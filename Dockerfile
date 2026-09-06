# ========== 基础镜像 ==========
FROM python:3.11-slim

WORKDIR /app

# 避免交互提示，加快安装
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

# 安装系统依赖（chromadb 等需要的基础库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先拷贝依赖文件以利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 拷贝源码
COPY . .

# 默认拷贝 .env.example 为 .env（正式部署请自行覆盖为真实密钥）
RUN if [ ! -f .env ]; then cp .env.example .env; fi

# 端口：8000 = FastAPI，8501 = Streamlit
EXPOSE 8000 8501

# 默认启动 FastAPI，可通过 docker-compose 覆盖为 streamlit 命令
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]