"""Streamlit 前端：文档上传 / 知识库管理 / 多轮流式问答（通过 SSE 消费后端）。"""
from __future__ import annotations

import json
import os
import uuid
from typing import Dict, List

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG 智能问答", page_icon="🤖", layout="wide")


# ---------- 会话状态 ----------
def _init_state() -> None:
    defaults: Dict[str, object] = {
        "messages": [],          # 前端展示的消息
        "history": [],           # 传给后端的多轮历史 [(q, a), ...]
        "domain": "企业知识库",
        "use_few_shot": True,
        "persona": "warm_guide",
        "user_id": "default",
        "session_id": str(uuid.uuid4()),  # 本次会话的稳定用户标识
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

client = httpx.Client(timeout=600)


def get_stats() -> dict:
    try:
        r = client.get(f"{API_BASE}/kb/stats")
        return r.json()
    except Exception:
        return {"count": 0, "sources": []}


# ---------- 侧边栏：知识库管理 ----------
with st.sidebar:
    st.title("📚 知识库")
    stats = get_stats()
    st.caption(f"当前片段数：**{stats.get('count', 0)}**")
    st.divider()

    st.subheader("上传文档")
    uploaded = st.file_uploader(
        "支持 .txt / .md / .pdf / .docx",
        type=["txt", "md", "markdown", "pdf", "docx"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        if st.button("🚀 写入知识库", use_container_width=True):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue())}
                r = client.post(f"{API_BASE}/ingest", files=files)
                if r.status_code == 200:
                    data = r.json()
                    st.success(f"已写入 {data['chunks']} 个片段：{data['source']}")
                else:
                    st.error(r.json().get("detail", "接入失败"))
            except Exception as e:
                st.error(f"上传失败: {e}")

    st.divider()
    st.subheader("已有文档")
    sources = stats.get("sources", [])
    if not sources:
        st.caption("知识库为空，请先上传文档。")
    for src in sources:
        col1, col2 = st.columns([4, 1])
        col1.caption(f"📄 {src}")
        if col2.button("🗑️", key=f"del_{src}", help=f"删除 {src}"):
            client.delete(f"{API_BASE}/kb/source/{src}")
            st.rerun()

    if sources and st.button("清空全部", use_container_width=True):
        client.delete(f"{API_BASE}/kb")
        st.rerun()

    st.divider()
    st.subheader("🧭 人设与记忆")
    persona_label = st.selectbox(
        "对话人设",
        options=["warm_guide", "professional", "companion"],
        format_func=lambda p: {
            "professional": "📊 专业严谨",
            "warm_guide": "💗 专业+温暖（推荐）",
            "companion": "🤗 情感陪伴",
        }[p],
        index=["warm_guide", "professional", "companion"].index(st.session_state.persona),
    )
    st.session_state.persona = persona_label
    use_q = st.text_input("用户标识（记忆隔离）", value=st.session_state.user_id)
    st.session_state.user_id = use_q or "default"
    if st.button("🧹 清除我的记忆", use_container_width=True):
        client.delete(f"{API_BASE}/memory/{st.session_state.user_id}")
        st.rerun()
    with st.expander("📖 查看记忆"):
        try:
            mem = client.get(f"{API_BASE}/memory/{st.session_state.user_id}").json()
            st.json({"画像": mem.get("profile"), "备注": mem.get("notes"), "最近互动": mem.get("history")})
        except Exception:
            st.caption("后端未启动或暂无记忆。")

    st.divider()
    st.session_state.domain = st.text_input("领域标签", value=st.session_state.domain)
    st.session_state.use_few_shot = st.checkbox(
        "启用 Few-Shot 示范", value=st.session_state.use_few_shot
    )

# ---------- 主区域：对话 ----------
st.title("🤖 RAG 垂直领域智能问答")
st.caption("自定义领域 · 上传知识库 · 多轮检索增强对话（DeepSeek 驱动）")

if "messages" not in st.session_state or not st.session_state.messages:
    st.info("👈 先在左侧上传文档到知识库，然后开始提问。")

# 渲染历史消息
_prev_sources: Dict[str, List[dict]] = st.session_state.get("_sources_map", {})

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i in _prev_sources and _prev_sources[i]:
            with st.expander("📎 检索到的资料"):
                for src in _prev_sources[i]:
                    st.markdown(f"**来源: {src['source']}** （相关度 {src['score']:.3f}）")
                    st.text(src["content"])

prompt = st.chat_input("想问点什么？")


def _stream_chat(question: str, history: List[List[str]]) -> tuple[str, list]:
    """调用后端 SSE 接口，边读边产出字符，返回 (answer, sources)。"""
    payload = {
        "question": question,
        "history": history,
        "domain": st.session_state.domain,
        "use_few_shot": st.session_state.use_few_shot,
        "persona": st.session_state.persona,
        "user_id": st.session_state.user_id,
    }
    collected: List[str] = []
    sources: list = []
    with client.stream("POST", f"{API_BASE}/chat/stream", json=payload) as resp:
        if resp.status_code != 200:
            return f"请求失败（{resp.status_code}）", []
        buffer = ""
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            if raw_line.startswith("event:"):
                buffer = raw_line[6:].strip()
                continue
            if raw_line.startswith("data:"):
                data = json.loads(raw_line[5:])
                if buffer == "delta":
                    chunk = data if isinstance(data, str) else ""
                    collected.append(chunk)
                    yield "".join(collected), sources
                elif buffer == "done":
                    answer = data if isinstance(data, str) else ""
                    collected.append(answer)
                elif buffer == "retrieved":
                    sources = data
                elif buffer == "error":
                    collected.append(f"[错误] {data.get('detail', '未知')}")
    return "".join(collected), sources


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer_ph = st.empty()
        idx = len(st.session_state.messages)  # assistant 消息即将加入
        answer = ""
        sources: list = []
        try:
            placeholder = st.empty()
            for partial, cur_sources in _stream_chat(prompt, st.session_state.history):
                placeholder.markdown(partial + "▌")
                answer, sources = partial, cur_sources
            placeholder.markdown(answer)
            _prev_sources[idx] = sources
        except Exception as e:
            placeholder.markdown(f"⚠️ 调用失败: {e}")

    # 落库多轮历史（仅保留最近 5 轮）
    st.session_state.history.append([prompt, answer])
    st.session_state.history = st.session_state.history[-10:]
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state["_sources_map"] = _prev_sources
    st.rerun()