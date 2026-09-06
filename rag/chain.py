"""RAG 主链路：文档接入 + 检索 + 生成 + 流式，统一对外接口。"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Generator, List, Sequence

from langchain_core.messages import BaseMessage

from config import settings
from rag.document_loader import SUPPORTED_EXTS, load_document
from rag.llm import get_llm, stream_chat
from rag.memory import MemoryStore
from rag.prompt_templates import build_messages
from rag.retriever import Retriever
from rag.text_splitter import split_documents
from rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    content: str
    source: str
    score: float


@dataclass
class RagResult:
    answer: str
    sources: List[RetrievedDoc] = field(default_factory=list)
    queried: bool = False


class RAGChain:
    """文档接入或在 one-stop 服务中复用 VectorStoreManager。"""

    def __init__(self, manager: VectorStoreManager | None = None) -> None:
        self.manager = manager or VectorStoreManager()
        self.retriever = Retriever(self.manager)
        self.llm = get_llm()
        self.memory = MemoryStore(settings.memory_dir)

    # ---------- 文档接入 ----------
    def ingest_file(self, file_path: str, doc_id: str | None = None) -> dict:
        """加载→切分→嵌入→入库。返回统计信息。"""
        docs = load_document(file_path)
        split = split_documents(docs)
        for d in split:
            d.metadata["doc_id"] = doc_id or os.path.basename(file_path)
        n = self.manager.add_documents(split)
        return {"chunks": n, "source": os.path.basename(file_path)}

    def ingest_files_from_dir(self, dir_path: str) -> dict:
        from rag.document_loader import load_directory

        docs = load_directory(dir_path)
        split = split_documents(docs)
        for d in split:
            d.metadata.setdefault("doc_id", d.metadata.get("source", "unknown"))
        n = self.manager.add_documents(split)
        return {"chunks": n, "sources": [d.metadata.get("source") for d in docs]}

    # ---------- 检索 ----------
    def retrieve(self, query: str, k: int | None = None) -> List[RetrievedDoc]:
        docs = self.retriever.retrieve(query, k=k)
        results: List[RetrievedDoc] = []
        scored = self.manager.similarity_search_with_score(query, k=len(docs))
        score_map = {d.page_content: s for d, s in scored}
        for doc in docs:
            results.append(
                RetrievedDoc(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "未知来源"),
                    score=score_map.get(doc.page_content, -1.0),
                )
            )
        return results

    # ---------- 生成 ----------
    def _run(
        self,
        question: str,
        history: Sequence[tuple[str, str]] | None,
        domain: str,
        include_few_shot: bool,
        persona_type: str | None = None,
        user_id: str | None = None,
    ) -> RagResult:
        rdocs = self.retrieve(question)
        memory_context = (
            self.memory.snapshot(user_id) if user_id else ""
        )
        messages = build_messages(
            question,
            [d for d in self._as_lc_docs(rdocs)],
            history=history,
            domain=domain,
            include_few_shot=include_few_shot,
            persona_type=persona_type,
            memory_context=memory_context,
        )
        answer = _invoke_once(self.llm, messages)
        if user_id:
            self.memory.push_exchange(user_id, question, answer)
        return RagResult(answer=answer, sources=rdocs, queried=True)

    def query(
        self,
        question: str,
        history: Sequence[tuple[str, str]] | None = None,
        domain: str = "企业知识库",
        include_few_shot: bool = True,
        persona_type: str | None = None,
        user_id: str | None = None,
    ) -> RagResult:
        """非流式完整问答。"""
        return self._run(
            question, history, domain, include_few_shot, persona_type, user_id
        )

    def stream(
        self,
        question: str,
        history: Sequence[tuple[str, str]] | None = None,
        domain: str = "企业知识库",
        include_few_shot: bool = True,
        persona_type: str | None = None,
        user_id: str | None = None,
    ) -> Generator[tuple[str, object], None, RagResult]:
        """流式问答。产出 (event, data) 事件，供 SSE 层转发；结束后返回完整结果。

        事件类型：retrieved / delta / done / error
        """
        rdocs = self.retrieve(question)
        yield ("retrieved", [{"content": d.content, "source": d.source, "score": d.score} for d in rdocs])

        memory_context = self.memory.snapshot(user_id) if user_id else ""
        messages = build_messages(
            question,
            self._as_lc_docs(rdocs),
            history=history,
            domain=domain,
            include_few_shot=include_few_shot,
            persona_type=persona_type,
            memory_context=memory_context,
        )

        answer_chunks: List[str] = []
        try:
            for chunk in stream_chat(self.llm, messages):
                answer_chunks.append(chunk)
                yield ("delta", chunk)
        except Exception as e:  # noqa: BLE001 - 由生成层兜底
            logger.error("问答生成异常: %s", e)
            yield ("error", str(e))

        answer = "".join(answer_chunks)
        if not answer.strip():
            answer = _invoke_once(self.llm, messages)
            yield ("delta", answer)
        yield ("done", answer)
        if user_id:
            self.memory.push_exchange(user_id, question, answer)
        return RagResult(answer=answer, sources=rdocs, queried=True)

    @staticmethod
    def _as_lc_docs(rdocs: Sequence[RetrievedDoc]):
        from langchain_core.documents import Document as LCDoc

        return [
            LCDoc(page_content=r.content, metadata={"source": r.source}) for r in rdocs
        ]


def _invoke_once(llm, messages: Sequence[BaseMessage]) -> str:
    """单次非流式调用（作为最终兜底）。"""
    try:
        res = llm.invoke(list(messages))
        return res.content if isinstance(res.content, str) else ""
    except Exception as e:  # noqa: BLE001
        logger.error("LLM 调用最终失败: %s", e)
        return "抱歉，大模型暂时不可用，请稍后重试。"


# ---------- 上传落盘工具 ----------
def save_upload(file_obj, upload_dir: str | None = None) -> str:
    """把上传的临时文件保存到 data/uploads 并返回路径。"""
    upload_dir = upload_dir or settings.upload_dir
    ext = os.path.splitext(file_obj.filename)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"不支持的文件类型 {ext}，支持 {sorted(SUPPORTED_EXTS)}")
    os.makedirs(upload_dir, exist_ok=True)
    dest = os.path.join(upload_dir, os.path.basename(file_obj.filename))
    with open(dest, "wb") as f:
        shutil.copyfileobj(file_obj.file, f)
    return dest