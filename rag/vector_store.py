"""ChromaDB 向量库管理：建库、写入、删除、查询、血缘追踪。"""
from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import settings
from rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

# Chroma 元数据仅允许标量，过滤掉 dict/list 等复杂值，避免入库报错
def _filter_metadata(documents: List[Document]) -> List[Document]:
    allowed = (str, int, float, bool, type(None))
    out: List[Document] = []
    for doc in documents:
        meta = {
            k: v
            for k, v in (doc.metadata or {}).items()
            if isinstance(v, allowed)
        }
        if meta != doc.metadata:
            doc = Document(page_content=doc.page_content, metadata=meta)
        out.append(doc)
    return out


class VectorStoreManager:
    """封装 Chroma 持久化存储，统一管理 collection。"""

    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: str | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.collection_name
        self.persist_dir = persist_dir or settings.chroma_dir
        self._store: Chroma | None = None

    def _get_store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=get_embeddings(),
                persist_directory=self.persist_dir,
            )
        return self._store

    # ---------- 写入 ----------
    def add_documents(self, documents: List[Document]) -> int:
        """向知识库写入文档块，返回写入条数。"""
        store = self._get_store()
        documents = _filter_metadata(documents)
        ids = [str(uuid.uuid4()) for _ in documents]
        store.add_documents(documents=documents, ids=ids)
        logger.info("已写入 %d 个片段到知识库 [%s]", len(ids), self.collection_name)
        return len(ids)

    # ---------- 查询 ----------
    def similarity_search(self, query: str, k: int | None = None) -> List[Document]:
        """Vector 相似度检索。"""
        store = self._get_store()
        return store.similarity_search(query, k=k or settings.top_k)

    def similarity_search_with_score(
        self, query: str, k: int | None = None
    ) -> List[tuple[Document, float]]:
        """返回 (Document, 相似度得分)。距离越近得分越低，调用方需换算。"""
        store = self._get_store()
        return store.similarity_search_with_relevance_scores(
            query, k=k or settings.top_k
        )

    # ---------- 元数据管理 ----------
    def get_sources(self) -> List[str]:
        """列出知识库中包含的文档来源（去重）。"""
        store = self._get_store()
        metas = store.get()["metadatas"] or []
        seen: dict[str, str] = {}
        for m in metas:
            src = (m or {}).get("source", "unknown")
            seen[src] = seen.get(src, "") or (m or {}).get("doc_id", "")
        return list(seen.keys())

    def get_doc_ids(self) -> List[str]:
        store = self._get_store()
        metas = store.get()["metadatas"] or []
        return [m.get("doc_id", "") for m in metas if m]

    def delete_by_source(self, source: str) -> int:
        """删除指定来源文档的所有片段。"""
        store = self._get_store()
        res = store.get(where={"source": source})
        ids = res.get("ids") or []
        if ids:
            store.delete(ids=ids)
            logger.info("已删除来源 [%s] 的 %d 个片段", source, len(ids))
        return len(ids)

    def delete_all(self) -> int:
        store = self._get_store()
        res = store.get()
        ids = res.get("ids") or []
        if ids:
            store.delete(ids=ids)
        logger.info("已清空知识库，删除 %d 个片段", len(ids))
        return len(ids)

    def count(self) -> int:
        store = self._get_store()
        return store._collection.count()

    @property
    def store(self) -> Chroma:
        return self._get_store()


_default_manager: Optional[VectorStoreManager] = None


def get_store_manager() -> VectorStoreManager:
    """进程级单例，避免重复初始化连接。"""
    global _default_manager
    if _default_manager is None:
        _default_manager = VectorStoreManager()
    return _default_manager