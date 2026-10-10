# 🤖 RAG 垂直领域智能问答系统

基于 **LangChain + DeepSeek（LLM）+ ChromaDB（向量库）** 从 0 搭建的垂直领域知识问答系统。支持**文档上传 → 知识库检索 → 多轮智能问答**的完整闭环，并通过 Prompt 工程抑制幻觉、SSE 流式响应、重试降级与人工评估测试集保障回答质量与系统稳定。

> 项目周期（`2025.07 – 2025.08`），核心难点：**缓解幻觉、提高回答准确率**。

---

## ✨ 核心特性

| 模块 | 技术方案 |
|------|----------|
| **文档加载** | TXT / Markdown / PDF / DOCX，标题感知切分 |
| **文本切分** | 递归字符切分 + Markdown 标题切分（`RecursiveCharacterTextSplitter`） |
| **向量嵌入** | 默认本地 `BAAI/bge-small-zh-v1.5`（也可切换远端 Embedding API） |
| **向量检索** | ChromaDB 持久化 + 相似度检索 + 得分阈值过滤 + n-gram 冗余去重 |
| **生成（LLM）** | DeepSeek（兼容 OpenAI），`max_retries` 重试 + 逐级降级 |
| **Prompt 工程** | Few-Shot Learning + 幻觉抑制护栏（强制“资料外拒答”） |
| **情感陪伴** | 三种人设（专业 / 专业+温暖 / 情感陪伴），自然口语表达、承接情绪 |
| **无“AI 味”** | 风格约束：不说“作为AI”、不堆砌列表、减少书面连接词、适当共情反问 |
| **记忆能力** | 跨会话长期记忆（用户画像/备注/最近互动），多轮历史压缩 |
| **交互** | SSE 流式响应，多轮对话 |
| **稳定性** | 超时 / 限流重试、整段输出兜底、错误事件透传 |
| **评估** | 人工评估测试集 + 指标（准确率 / 拒答率 / 幻觉率）+ A/B 对比 |

---

## 📁 项目结构

```
rag-qa-system/
├── app.py                    # Streamlit 前端（上传/知识库/多轮流式对话）
├── api.py                    # FastAPI 后端（含 SSE 流式问答）
├── config.py                 # 全局配置（支持 .env 覆盖）
├── requirements.txt          # 依赖
├── Dockerfile                # 容器化
├── docker-compose.yml        # 一键起 API + UI
├── rag/
│   ├── document_loader.py    # 文档加载
│   ├── text_splitter.py      # 文本切分（含标题感知）
│   ├── embeddings.py         # 向量嵌入工厂（HF / 远端）
│   ├── vector_store.py       # ChromaDB 向量库
│   ├── retriever.py          # 检索：召回+过滤+去重
│   ├── prompt_templates.py   # 系统提示词 + Few-Shot + 幻觉护栏 + 人设/风格
│   ├── llm.py                # DeepSeek 封装：重试 + 降级
│   ├── memory.py             # 跨会话长期记忆（画像/备注/历史）
│   └── chain.py              # RAG 主链路编排（含流式 + 记忆）
├── evaluation/
│   ├── evalset.py            # 人工评估测试集
│   ├── metrics.py            # 指标计算
├── scripts/
│   ├── ingest.py             # CLI 文档接入
│   └── run_eval.py           # 评估 + A/B 对比
├── docs/sample_employee_handbook.md   # 示例知识库文档
└── tests/test_basic.py       # 单元测试
```

---

## 🚀 快速开始

### 1. 准备环境

```bash
# Python 3.11+
pip install -r requirements.txt
```

### 2. 配置 DeepSeek 密钥

```bash
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY
```

> 首次运行时，系统会下载中文向量模型 `bge-small-zh-v1.5`（约 100MB，仅一次）。
> 如国内网络拉取模型失败，可在 `.env` 配置 `HTTPS_PROXY`/`HTTP_PROXY`。

### 3. 导入示例知识库

```bash
python scripts/ingest.py docs/sample_employee_handbook.md
```

### 4. 启动服务

**终端 A：FastAPI 后端（含 SSE 接口）**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**终端 B：Streamlit 前端**
```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`，即可上传你的领域文档并开始多轮问答。

### 5. 运行评估

```bash
# 在知识库中导入示例文档后
python scripts/run_eval.py          # 输出准确率/拒答率/幻觉率
python scripts/run_eval.py --ab     # A/B 对比 few-shot 开关 与 TOP_K 灵敏性
python scripts/run_eval.py --output out.json
```

---

## 🐳 Docker 部署

```bash
# 先配置好 .env（含 DEEPSEEK_API_KEY）
docker compose up --build
```

- API: `http://localhost:8000`（`/docs` 查看 Swagger）
- UI: `http://localhost:8501`

---

## 🔌 API 速览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/ingest` | 上传文档写入知识库（multipart） |
| GET | `/kb/stats` | 知识库片段数 / 来源列表 |
| GET | `/kb/sources` | 来源列表 |
| DELETE | `/kb/source/{source}` | 删除某来源 |
| DELETE | `/kb` | 清空知识库 |
| POST | `/chat` | 非流式问答 |
| POST | `/chat/stream` | **SSE 流式问答** |
| GET | `/memory/{user_id}` | 查看某用户长期记忆 |
| POST | `/memory/{user_id}` | 写入用户画像/备注 |
| DELETE | `/memory/{user_id}` | 清除某用户记忆 |
| GET | `/health` | 健康检查 |

### SSE 事件格式

```
event: retrieved   # 检索到的片段（附带来源与相关度）
event: delta       # 生成中的字符块
event: done        # 完整回答
event: error       # 出错信息
```

---

## 🧠 抑制幻觉的设计要点

1. **双保险提示**：系统提示词要求“只依据资料回答，资料外明确拒答”；Few-Shot 示范提供了“未命中时如何拒答”的例子，让模型学到行为范式。
2. **检索质量门控**：相似度低于阈值 `SCORE_THRESHOLD` 的片段会被丢弃；冗余片段去重，避免上下文被重复内容污染。
3. **归因可追溯**：回答基于带来源标注的资料片段，前端可展开查看“检索到的资料”，人工可核验。
4. **评估量化**：测试集中含 `unanswerable` 类别样本，专门测量模型在“资料未覆盖”时是否拒答、是否编造数字（幻觉率）。

---

## 💗 情感陪伴、无“AI 味”与记忆

- **人设切换**：通过 `PERSONA_TYPE`（或在界面上选择）在 `professional`（专业严谨）/ `warm_guide`（专业+温暖，默认）/ `companion`（情感陪伴优先）之间切换，Prompt 中内置对应的语气与共情策略。
- **无“AI 味”**：风格约束明确要求模型不说“作为AI”、不以客套话开头、不堆砌列表、不使用“综上所述”等书面连接词，并鼓励承接语气、适当反问共情。
- **长期记忆**：跨会话记住用户画像、偏好与最近互动（`data/memory/{user_id}.json`）。界面可切换“用户标识”实现多用户记忆隔离，也可一键清除记忆。记忆通过提示词注入，让回答更个性化、更像真人。
- **克制与护栏并存**：陪伴风格主要影响“怎么说”，不放松“资料未覆盖必须拒答”的幻觉护栏，保证陪陪伴时也不会胡说。

---

## 📋 备注

- 嵌入默认使用本地模型，无需额外密钥；如需调用远端 Embedding，可在 `.env` 把 `EMBEDDING_PROVIDER` 改为 `openai_compatible` 并配置对应的 `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY`。
- 生产部署建议：将 `chroma_db/` 与 `data/uploads/` 挂载到持久化卷，并妥善保管 `.env`。

---

## 🔌 相关项目：DSH 会话回退插件

顺带一提，我还写了个小工具 **[dsh-rewind](https://github.com/LUDIWUSI12138/dsh-sqy-rewind)**：
给 DSH（DeepSeek Harness）桌面版用的会话回退插件，把当前会话退回某一轮对话发起之前 ——
模型历史就地遮蔽，界面同步隐藏被回退的轮次。

与上面的 RAG 项目没有依赖关系，已独立成仓库维护。

## 📄 License

MIT