"""DeepSeek LLM 封装：OpenAI 兼容调用 + 指数退避重试 + 逐级降级保障。"""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config import settings

logger = logging.getLogger(__name__)

# 启动即校验 API Key，避免运行到一半才发现
if not settings.deepseek_api_key:
    raise RuntimeError(
        "缺少 DEEPSEEK_API_KEY。请在项目根目录创建 .env（参考 .env.example）后重试。"
    )


def get_llm() -> BaseChatModel:
    """返回配置好的 DeepSeek ChatOpenAI 客户端，带超时与重试。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
        # 显式关闭 langchain 内部的 __call__ 缓存
        cache=False,
        streaming=True,
    )


def stream_chat(llm: BaseChatModel, messages: Sequence[BaseMessage]) -> Iterable[str]:
    """对流式输出做稳定的迭代，丢出时可被上层降级捕获。"""
    try:
        for chunk in llm.stream(messages):
            if not chunk or not chunk.content:
                continue
            yield chunk.content if isinstance(chunk.content, str) else str(chunk.content)
    except Exception as e:  # noqa: BLE001 - 统一在生成层做降级
        logger.warning("流式生成中断: %s，尝试降级为整段输出", e)
        yield _fallback_full_completion(llm, messages)


def _fallback_full_completion(llm: BaseChatModel, messages: Sequence[BaseMessage]) -> str:
    """降级方案：改用一次性调用（invoke）获取完整回答。"""
    try:
        res = llm.invoke(messages)
        return res.content if isinstance(res.content, str) else ""
    except Exception as e:  # noqa: BLE001
        logger.error("LLM 整体降级失败: %s", e)
        return (
            "[系统] 抱歉，大模型暂时不可用（网络或限流原因）。"
            "请稍后重试。检索到的相关片段见下方参考内容。"
        )