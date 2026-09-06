"""FastAPI 后端：文档上传接入 + 知识库管理 + RAG 问答（含 SSE 流式）。"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import settings
from rag.chain import RAGChain, save_upload
from rag.memory import get_memory_store
from rag.vector_store import get_store_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(title="RAG 垂直领域智能问答 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_chain: Optional[RAGChain] = None


def get_chain() -> RAGChain:
    global _chain
    if _chain is None:
        _chain = RAGChain(manager=get_store_manager())
    return _chain


# ---------- 数据模型 ----------
class ChatRequest(BaseModel):
    question: str
    history: List[List[str]] = []  # [问题, 回答] 的多轮历史
    domain: str = "企业知识库"
    use_few_shot: bool = True
    # 人设：professional | warm_guide | companion
    persona: str = ""
    # 用户标识：用于长期记忆（不同用户记忆隔离）
    user_id: str = ""


class IngestRequest(BaseModel):
    path: str


@app.get("/health")
def health():
    return {"status": "ok", "chunks": get_chain().manager.count()}


# ---------- 文档接入 ----------
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """上传文档并写入知识库。"""
    try:
        path = save_upload(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        info = get_chain().ingest_file(path)
    except Exception as e:  # noqa: BLE001
        logger.exception("doc ingest failed")
        raise HTTPException(status_code=500, detail=f"文档接入失败: {e}")
    return {"ok": True, **info}


@app.post("/ingest/path")
async def ingest_by_path(req: IngestRequest):
    info = get_chain().ingest_file(req.path)
    return {"ok": True, **info}


@app.get("/kb/sources")
def kb_sources():
    return {"sources": get_chain().manager.get_sources()}


@app.get("/kb/stats")
def kb_stats():
    m = get_chain().manager
    return {"count": m.count(), "sources": m.get_sources()}


@app.delete("/kb/source/{source}")
def delete_source(source: str):
    n = get_chain().manager.delete_by_source(source)
    return {"ok": True, "deleted": n}


@app.delete("/kb")
def clear_kb():
    n = get_chain().manager.delete_all()
    return {"ok": True, "deleted": n}


# ---------- 长期记忆 ----------
class MemoryUpdate(BaseModel):
    profile: dict = {}
    note: str = ""


@app.get("/memory/{user_id}")
def get_memory(user_id: str):
    store = get_memory_store()
    data = store.load(user_id)
    return {"user_id": user_id, **data}


@app.post("/memory/{user_id}")
def update_memory(user_id: str, req: MemoryUpdate):
    store = get_memory_store()
    for k, v in req.profile.items():
        store.remember(user_id, k, v)
    if req.note:
        store.add_note(user_id, req.note)
    return {"ok": True}


@app.delete("/memory/{user_id}")
def clear_memory(user_id: str):
    get_memory_store().clear(user_id)
    return {"ok": True, "cleared": user_id}


# ---------- RAG 问答 ----------
@app.post("/chat")
def chat(req: ChatRequest):
    """非流式问答。"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    history = [(h[0], h[1]) for h in req.history if len(h) == 2]
    uid = req.user_id or settings.default_user_id
    result = get_chain().query(
        req.question,
        history=history,
        domain=req.domain,
        include_few_shot=req.use_few_shot,
        persona_type=req.persona or None,
        user_id=uid,
    )
    return {
        "answer": result.answer,
        "sources": [
            {"content": s.content, "source": s.source, "score": round(s.score, 4)}
            for s in result.sources
        ],
    }


def sse_format(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE 流式问答。

    事件序列：
      event: retrieved  → 检索到的片段
      event: delta      → 生成中的字符块
      event: done       → 完整回答
      event: error      → 出错信息
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    history = [(h[0], h[1]) for h in req.history if len(h) == 2]
    uid = req.user_id or settings.default_user_id

    def gen():
        try:
            for event, data in get_chain().stream(
                req.question,
                history=history,
                domain=req.domain,
                include_few_shot=req.use_few_shot,
                persona_type=req.persona or None,
                user_id=uid,
            ):
                yield sse_format(event, data)
        except Exception as e:  # noqa: BLE001
            logger.exception("stream chat error")
            yield sse_format("error", {"detail": str(e)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=settings.host, port=settings.port, reload=True)