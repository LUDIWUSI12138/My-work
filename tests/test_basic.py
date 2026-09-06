"""基础测试：Prompt 拼装与切分逻辑（不依赖网络/Key）。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from rag.text_splitter import split_documents
from rag.prompt_templates import build_messages, build_system_message


def test_system_prompt_contains_hallucination_guard():
    msg = build_system_message(domain="企业知识库")
    text = msg.content
    assert "不得编造" in text
    assert "资料里没有提到" in text or "资料中未提及" in text


def test_build_messages_structure():
    docs = [Document(page_content="年假：满 1 年不满 10 年 5 天", metadata={"source": "手册"})]
    msgs = build_messages(
        "年假多少天？",
        docs,
        history=[("你好", "你好，有什么可以帮你？")],
        domain="企业知识库",
    )
    assert msgs[0].type == "system"
    # 系统 + 4 条 few-shot + 1 条当前问题
    assert len(msgs) == 1 + 4 + 1
    assert "年假" in msgs[-1].content


def test_split_long_document():
    text = "。" .join(["这是一段用于测试切分的句子句子句子"] * 300)
    docs = split_documents([Document(page_content=text, metadata={"source": "a.txt"})], chunk_size=200, chunk_overlap=40)
    assert len(docs) >= 2
    for d in docs:
        assert len(d.page_content) <= 200 + 40


def test_markdown_header_split():
    md = "# 第一章\n内容一内容一内容一\n## 第一节\n内容二内容二"
    docs = split_documents([Document(page_content=md, metadata={"source": "a.md"})], chunk_size=200, chunk_overlap=40)
    # 至少按标题切分成多块
    assert len(docs) >= 2