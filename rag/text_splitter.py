"""文本切分：递归字符切分 + 基于标题的感知切分。

递归切分可处理长文档；对 markdown 文档自动启用标题感知（按标题分节，避免切断语义块）。
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from config import settings

DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]

# Markdown 标题层级，用于标题感知切分
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]


def _split_markdown_by_headers(document: Document, chunk_size: int, overlap: int) -> List[Document]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    try:
        sections = splitter.split_text(document.page_content)
    except Exception:  # noqa: BLE001 标题解析失败则回退到普通切分
        return _split_plain(document, chunk_size, overlap)

    # 纯文本切片器用于二次切分（标题下的长文本）
    refine = RecursiveCharacterTextSplitter(
        separators=DEFAULT_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    out: List[Document] = []
    for sec in sections:
        # Chroma 元数据要求标量/字符串，把标题层级拼为扁平字符串
        flat_headers = " > ".join(
            str(v) for k, v in (sec.metadata or {}).items() if v
        )
        meta = {**document.metadata}
        if flat_headers:
            meta["headers"] = flat_headers
        else:
            meta.pop("headers", None)
        if len(sec.page_content) <= chunk_size:
            sec.metadata = meta
            out.append(sec)
        else:
            for sub in refine.split_documents([sec]):
                sub.metadata = meta
                out.append(sub)
    return out


def _split_plain(document: Document, chunk_size: int, overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=DEFAULT_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_documents([document])


def split_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    use_token: bool = False,
) -> List[Document]:
    """按配置切分文档。use_token=True 时使用基于 token 的切分（估算更稳定）。"""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    result: List[Document] = []
    for doc in documents:
        source = doc.metadata.get("source", "").lower()
        if use_token:
            splitter = TokenTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            result.extend(splitter.split_documents([doc]))
        elif source.endswith((".md", ".markdown")):
            result.extend(_split_markdown_by_headers(doc, chunk_size, chunk_overlap))
        else:
            result.extend(_split_plain(doc, chunk_size, chunk_overlap))
    return result