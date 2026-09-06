"""全局配置：支持环境变量覆盖，自带合理默认值。"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

# 就近加载 .env（放在项目根目录即可）
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
MEMORY_DIR = os.environ.get("MEMORY_DIR", os.path.join(DATA_DIR, "memory"))
CHROMA_DIR = os.environ.get("CHROMA_DIR", os.path.join(BASE_DIR, "chroma_db"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "rag_kb")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)


class Paths:
    """仅承载路径，方便 import 而不触发重依赖。"""

    upload_dir = UPLOAD_DIR
    memory_dir = MEMORY_DIR


class Settings:
    # ---- LLM: DeepSeek ----
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", 0.2))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", 1024))
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", 60.0))
    max_retries: int = int(os.getenv("MAX_RETRIES", 3))

    # ---- Embedding ----
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

    # ---- 人设与记忆 ----
    # professional | warm_guide | companion
    persona_type: str = os.getenv("PERSONA_TYPE", "warm_guide")
    # 默认用户标识（未指定 user_id 时）
    default_user_id: str = os.getenv("DEFAULT_USER_ID", "default")

    # ---- Chunking ----
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 80))

    # ---- Retrieval ----
    top_k: int = int(os.getenv("TOP_K", 4))
    score_threshold: float = float(os.getenv("SCORE_THRESHOLD", 0.3))
    # 压缩/过滤低相关片段的比例（0 表示关闭）
    dedup_overlap_ratio: float = float(os.getenv("DEDUP_OVERLAP_RATIO", 0.8))

    # ---- 会话 ----
    max_history_rounds: int = int(os.getenv("MAX_HISTORY_ROUNDS", 4))

    # ---- 服务 ----
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8000))
    upload_dir: str = UPLOAD_DIR
    memory_dir: str = MEMORY_DIR
    chroma_dir: str = CHROMA_DIR
    collection_name: str = COLLECTION_NAME


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()