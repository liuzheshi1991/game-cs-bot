# 游戏客服机器人 RAG Demo（Python）

这个项目提供一套可直接起步的客服机器人知识库方案：

- 原始业务文档：`docs/game_product_doc.md`
- 原始 FAQ 文档：`docs/customer_faq.md`
- RAG 脚手架（Python）：文档切分、向量入库、检索问答
- Web UI 界面：支持文本检索、向量检索、混合检索三种模式

## 1. 技术栈

- Python 3.10+
- Flask（Web 框架）
- ChromaDB（本地向量数据库）
- Sentence Transformers（中文向量模型）
- Ollama（推荐本地调试，用于生成自然语言答案）
- OpenAI API（可选）

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 构建知识库（支持两种方式）

```bash
python scripts/build_kb.py --mode both
```

可选模式：

- `--mode text`：仅构建文本检索索引（关键词召回）
- `--mode embedding`：仅构建 embedding 向量索引（Ollama embedding）
- `--mode both`：同时构建两种索引（推荐）

## 4. 进行问答

### 4.1 命令行模式

```bash
python scripts/ask.py "充值成功但钻石没到账怎么办？" --mode text
python scripts/ask.py "充值成功但钻石没到账怎么办？" --mode embedding
```

### 4.2 Web UI 界面（推荐）

启动 Flask Web 服务器：

```bash
python app.py
```

启动成功后访问：
- 本地：http://127.0.0.1:5000
- 局域网：http://your-ip:5000

#### UI 功能特性：

1. **三种检索模式切换**：
   - 文本检索：基于关键词匹配
   - 向量检索：基于语义相似度
   - 混合检索（默认）：同时使用两种方式，选择最优结果

2. **实时对话**：支持流式响应，打字机效果展示回复

3. **推荐问题**：当匹配度不够高时，自动推荐相关的标准问题

#### UI 界面截图：

![客服机器人UI界面](https://github.com/liuzheshi1991/game-cs-bot/assets/xxx/xxxx)

界面包含：
- 顶部标题栏
- 检索模式选择按钮
- 聊天消息区域（支持滚动）
- 底部输入框和发送按钮

## 5. 两种检索方式对比脚本

```bash
python scripts/compare_retrievals.py "充值成功但钻石没到账怎么办？" --top-k 4
```

该脚本会同时输出：

- text 检索的回答和召回上下文
- embedding 检索的回答和召回上下文

## 6. 使用 Ollama 做本地 RAG 调试（推荐）

1) 安装并启动 Ollama（默认端口 `11434`）  
2) 拉取模型（示例）：

```bash
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

3) 复制 `.env.example` 为 `.env`，确认：

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_EMBED_MODEL=nomic-embed-text
```

### 可选：切换到 OpenAI

在 `.env` 中设置：

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=你的key
OPENAI_MODEL=gpt-4o-mini
```

## 7. 项目结构

```text
docs/                    # 原始知识文档（可持续迭代）
scripts/build_kb.py      # 知识库构建脚本
scripts/ask.py           # 问答脚本
scripts/compare_retrievals.py  # 文本/embedding 对比脚本
src/rag/config.py        # 路径与参数配置
src/rag/loader.py        # 文档加载与切分
src/rag/indexer.py       # 构建 text/embedding 索引
src/rag/retriever.py     # text/embedding 检索
src/rag/qa.py            # RAG 问答流程
src/rag/embeddings.py    # 向量嵌入处理
data/kb_chunks.json      # 文本检索知识库
data/kb_vectors.json     # embedding 检索知识库
app.py                   # Flask Web 应用
templates/chat.html      # 聊天界面模板
```

## 8. 下一步建议

- 补充真实客服工单（脱敏）到 `docs/`，每周重建一次知识库
- 给 FAQ 增加标签字段（账号/充值/处罚）提升检索准确度
- 增加评测集（50-100 条问答）做召回率和准确率评估
- 添加用户认证和会话管理功能
- 部署到生产环境（使用 Gunicorn + Nginx）
