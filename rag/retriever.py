"""检索器：多路召回 + 得分过滤 + 冗余去重，保证注入上下文的质量。"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document

from config import settings
from rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class Retriever:
    """对向量库执行检索，并做质量过滤与去重。"""

    def __init__(self, manager: VectorStoreManager) -> None:
        self.manager = manager

    def _dedup(self, docs: List[Document]) -> List[Document]:
        """去除内容高度重复的片段（按 char-ngram 重叠度）。"""
        if settings.dedup_overlap_ratio <= 0:
            return docs

        def normalize(text: str) -> set:
            return set(text)

        kept: List[Document] = []
        for doc in docs:
            cur = normalize(doc.page_content)
            dup = False
            for prev in kept:
                prev_set = normalize(prev.page_content)
                if prev_set:
                    overlap = len(cur & prev_set) / len(prev_set)
                    if overlap > settings.dedup_overlap_ratio:
                        dup = True
                        break
            if not dup:
                kept.append(doc)
        return kept

    def retrieve(self, query: str, k: int | None = None, threshold: float | None = None) -> List[Document]:
        """召回→过滤→去重，返回排序后的片段列表（按相关性降序）。"""
        k = k or settings.top_k
        threshold = settings.score_threshold if threshold is None else threshold

        results = self.manager.similarity_search_with_score(query, k=k * 2)
        # relevance_scores 越大越相关；低于阈值视为不相关，剔除
        filtered: List[tuple[Document, float]] = [
            (doc, score) for doc, score in results if score >= threshold
        ]
        # 兜底：即使全部低于阈值，也保留相关性最高的一条，避免空检索
        if not filtered and results:
            top = max(results, key=lambda x: x[1])
            filtered = [top]

        filtered.sort(key=lambda x: x[1], reverse=True)
        docs = [d for d, _ in filtered[:k]]
        docs = self._dedup(docs)
        return docs