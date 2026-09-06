"""向量嵌入工厂：按配置选择本地 HF 模型或远端 OpenAI 兼容 Embedding 服务。"""
from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """缓存单例的 Embeddings 实例，避免重复加载模型。"""
    provider = settings.embedding_provider.lower()

    if provider == "openai_compatible":
        if not settings.embedding_api_key:
            raise ValueError(
                "embedding_provider=openai_compatible 但未配置 EMBEDDING_API_KEY"
            )
        from langchain_openai import OpenAIEmbeddings

        logger.info("使用远端 Embedding: %s", settings.embedding_model_name)
        return OpenAIEmbeddings(
            model=settings.embedding_model_name,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url or None,
        )

    # 默认 huggingface 本地模型
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info(
            "加载本地 Embedding 模型: %s（首次运行需下载，请耐心等待）", settings.embedding_model
        )
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError as e:
        raise RuntimeError(
            "未安装 langchain-huggingface / sentence-transformers，请执行 "
            "pip install -r requirements.txt；或改用 EMBEDDING_PROVIDER=openai_compatible"
        ) from e