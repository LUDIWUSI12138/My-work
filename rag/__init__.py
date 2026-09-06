"""RAG 智能问答系统核心包：文档加载 → 切分 → 嵌入 → 检索 → 生成。

为避免引入重量级依赖，这里不做顶层重导出；请按需直接导入子模块：
    from rag.chain import RAGChain
    from rag.prompt_templates import build_messages
"""
__version__ = "1.0.0"