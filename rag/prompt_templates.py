"""Prompt 工程：系统提示词（含幻觉抑制护栏） + Few-Shot 示范 + 上下文拼装。"""
from __future__ import annotations

from typing import List, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from config import settings

# ---------- 人物 / 陪伴风格预设 ----------
# professional: 专业严谨；warm_guide: 专业+温暖自然；companion: 情感陪伴优先
PERSONAS = {
    "professional": {
        "name": "智能知识助手",
        "style": (
            "保持专业、清晰、严谨的语调；结构化输出，分点清晰，直接命中问题。\n"
        ),
    },
    "warm_guide": {
        "name": "贴心知识向导",
        "style": (
            "在专业可靠的基础上，语气自然温暖、有人情味；用自然口语表达结论，"
            "适当承接用户情绪与语气，但知识准确性与幻觉抑制要求不变。\n"
        ),
    },
    "companion": {
        "name": "陪伴知识伙伴",
        "style": (
            "以情感陪伴为主、知识服务为辅。主动关心用户感受，说话亲切自然、像朋"
            "友般回应；当用户是在倾诉/闲聊或表露情绪时，优先回应情绪、给予陪伴，"
            "再视需要调用知识库辅助。回答不机械、不堆砌要点，简短走心。\n"
        ),
    },
}

NATURAL_DIRECTION = (
    "## 语言风格要求（让表达像真人、无“AI 味”）：\n"
    "1. 不要自称“作为一名AI/作为AI助手”，不要以“我希望/希望可以帮到你”等客套开头。\n"
    "2. 除非真的要分点，否则别用编号列表堆砌；能用一句自然的话说完就不要拆成三点。\n"
    "3. 少用“综上所述”“总的来说”“以上便是”这类书面连接词，直接说结果。\n"
    "4. 适当使用反问、共情和口语化表达（“听起来……” “是不是……”“其实……”），"
    "保持语气的温度，但避免过度油腻或过度使用网络流行语。\n"
    "5. 字数控制在能讲清楚即可：简单问题简短答，复杂问题清楚但不冗长。\n\n"
)

# ---------- 系统提示词：角色 + 幻觉抑制护栏 + 风格 ----------
SYSTEM_TEMPLATE = """你是「{domain}」领域的「{persona_name}」，基于给定的知识库资料回答用户问题。

## 回答规则（务必严格遵守）
1. 只能依据下方【参考资料】中的内容回答，不得编造知识库中不存在的信息。
2. 若【参考资料】不足以回答，请明确说出“资料里没有提到”，并给出可能的检索建议，绝不要猜测。
3. 若【参考资料】之间存在冲突，优先采信来源标注更权威或表述更具体的资料，并如实说明冲突。
4. 涉及的数据/数字/流程必须与资料原文一致，不得自行推算。
5. 对不确定的部分，用“通常/可能/据资料”等措辞加以限定，不做绝对化断言。

{persona_style}
{NATURAL_DIRECTION}
## 输出要求
先用自然的一句话给出核心结论；再根据需要补充依据（可标注来源）；最后视情况给出延伸或陪伴式回应。"""


# ---------- Few-Shot 示范：给定含不确定/未命中场景，引导模型正确拒答 ----------
FEW_SHOT_EXAMPLES: List[dict] = [
    {
        "context": "资料：\n[来源: 员工手册] 年假标准：工龄满 1 年不满 10 年，年假 5 天。",
        "question": "工龄 15 年的员工年假多少天？",
        "answer": "资料中未提及工龄 10 年以上员工的年假标准。根据资料，仅能确认“工龄满 1 年不满 10 年，年假 5 天”。建议向 HR 咨询或补充对应制度文件。",
    },
    {
        "context": "资料：\n[来源: 产品FAQ] 标准版最大并发连接数为 500；企业版支持 5000。",
        "question": "标准版限不限并发？",
        "answer": "根据资料，标准版最大并发连接数为 500，企业版为 5000。",
    },
]


def build_system_message(domain: str = "企业知识库", persona_type: str | None = None) -> SystemMessage:
    persona_type = persona_type or settings.persona_type
    persona = PERSONAS.get(persona_type, PERSONAS["warm_guide"])
    return SystemMessage(
        content=SYSTEM_TEMPLATE.format(
            domain=domain,
            persona_name=persona["name"],
            persona_style=persona["style"],
            NATURAL_DIRECTION=NATURAL_DIRECTION,
        )
    )


def _format_docs(docs: Sequence) -> str:
    """把检索到的片段拼成带来源标注的资料文本。"""
    if not docs:
        return "（未检索到相关资料）"
    lines = []
    for i, doc in enumerate(docs, 1):
        s = doc.metadata.get("source", "未知来源")
        header = ""
        h = doc.metadata.get("headers")
        if h:
            header = f"{h} — "
        lines.append(f"[片段{i} | 来源: {s}]{header}\n{doc.page_content}")
    return "\n\n".join(lines)


def _format_history(history: Sequence[tuple[str, str]], max_rounds: int) -> str:
    """把多轮对话历史拼装成上下文，仅保留最近 N 轮避免超长。"""
    if not history:
        return ""
    recent = history[-max_rounds:]
    turns = []
    for q, a in recent:
        turns.append(f"用户：{q}\n助手：{a}")
    return "\n".join(turns)


def build_messages(
    question: str,
    docs: Sequence,
    history: Sequence[tuple[str, str]] | None = None,
    domain: str = "企业知识库",
    include_few_shot: bool = True,
    persona_type: str | None = None,
    memory_context: str = "",
) -> List[BaseMessage]:
    """构建送到 LLM 的消息序列：系统 + 记忆 + Few-shot + 历史 + 当前问题+资料。"""
    history = history or []
    messages: List[BaseMessage] = [build_system_message(domain, persona_type)]

    # 注入长期记忆（跨会话画像/备注/历史互动），用于陪伴与个性化
    if memory_context and memory_context.strip() and memory_context != "（暂无记忆）":
        messages.append(
            HumanMessage(
                content=f"【关于这位用户，你记得】\n{memory_context}\n\n"
                "请自然地把记得的信息用在回应里（例如称呼、延续话题、体察情绪）。"
            )
        )

    if include_few_shot:
        for ex in FEW_SHOT_EXAMPLES:
            messages.append(HumanMessage(content=f"{ex['context']}\n\n问题：{ex['question']}"))
            messages.append(HumanMessage(content=f"回答：{ex['answer']}"))

    context = _format_docs(docs)
    hist = _format_history(history, max_rounds=5)

    user_content = f"""【参考资料】
{context}

【对话历史】
{hist or '（无）'}

请回答当前问题：
{question}"""
    messages.append(HumanMessage(content=user_content))
    return messages