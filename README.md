# MyAIHub · FastAPI AI 网关服务

一个基于 FastAPI 的轻量级 AI 网关，封装 DeepSeek / Ollama 的 OpenAI 兼容接口，提供 **非流式** 与 **流式（SSE）** 两种聊天能力，并内置全局异常处理与请求日志中间件。

## ✨ 特性

- 🤖 **多后端支持**：DeepSeek（云端）/ Ollama（本地），通过 `.env` 一键切换
- 🌊 **流式输出**：基于 SSE（Server-Sent Events）实现打字机效果
- 🛡️ **全局异常处理**：对标 Spring Boot `@ControllerAdvice`，统一响应格式
- 📝 **日志中间件**：记录请求路径、方法、脱敏请求体、状态码、耗时，输出至控制台 + 滚动文件
- 🧪 **pytest 覆盖**：正常、参数校验、业务异常、系统异常等多场景测试
- 🧩 **纯 ASGI 中间件**：不缓冲 `StreamingResponse`，保证流式响应真正实时

## 🏗️ 架构

```
┌──────────────┐   POST /ai/chat         ┌────────────────┐
│  Client      │ ──────────────────────► │                │
│  (Spring     │                          │   FastAPI      │
│   Boot /     │   POST /ai/chat/stream   │   (本服务)      │
│   浏览器)     │ ──────────────────────► │                │
└──────────────┘                          └───────┬────────┘
                                                  │  AsyncOpenAI
                                                  ▼
                                        ┌────────────────────┐
                                        │  DeepSeek / Ollama │
                                        └────────────────────┘
```

## 📁 项目结构

```
MyAIHub/
├── .env                       # 环境变量（API Key 等，不提交 Git）
├── requirements.txt
├── run.py                     # 快速启动脚本
├── sse_test.html              # 流式测试页面
├── logs/
│   └── app.log                # 滚动日志（自动生成）
├── tests/
│   ├── conftest.py
│   └── test_api.py
└── app/
    ├── main.py                # 应用入口：CORS / 异常处理器 / 中间件
    ├── api/
    │   └── chat.py            # /ai/chat 与 /ai/chat/stream
    ├── core/
    │   ├── config.py          # 配置（支持 DeepSeek / Ollama 切换）
    │   ├── ai_client.py       # AI 客户端封装（非流式 + 流式）
    │   ├── exceptions.py      # 自定义业务异常 BizException
    │   ├── logging_config.py  # 日志初始化
    │   ├── exception_handlers.py  # 全局异常处理器
    │   └── middleware.py      # 纯 ASGI 日志中间件
    └── models/
        ├── schemas.py
        └── response.py        # 统一响应格式 APIResponse
```

## 🚀 快速开始

### 1. 环境准备

- Python 3.10+
- （可选）Ollama，本地运行模型：`ollama pull qwen2.5:1.5b`

### 2. 安装依赖

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. 配置环境变量

在项目根目录创建 `.env`：

```ini
# 选择 AI 服务：deepseek 或 ollama
AI_SERVICE_TYPE=ollama

# DeepSeek 配置（当 AI_SERVICE_TYPE=deepseek 时生效）
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
AI_MODEL=deepseek-chat

# Ollama 配置（当 AI_SERVICE_TYPE=ollama 时生效）
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:1.5b
```

### 4. 启动服务

```bash
uvicorn app.main:app --reload
```

服务启动后：

- Swagger 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/

## 🔌 API

### `POST /ai/chat`（非流式）

**请求：**

```json
{ "prompt": "你好" }
```

**响应（成功）：**

```json
{ "reply": "你好！我是..." }
```

**响应（业务异常）：**

```json
{ "code": 40001, "message": "prompt 不能为空", "data": null }
```

### `POST /ai/chat/stream`（流式 SSE）

**请求：**

```json
{ "prompt": "你好" }
```

**响应（`text/event-stream`）：**

```
data:{"type":"message","content":"你"}
data:{"type":"message","content":"好"}
data:{"type":"message","content":"！"}
data:{"type":"done"}
```

**错误事件：**

```
data:{"type":"error","code":40001,"message":"prompt 不能为空"}
```

### 业务错误码约定

| code | 含义 |
|------|------|
| 40001 | prompt 为空 |
| 40002 | prompt 含非法关键字 |
| 500 | 服务端内部错误 |

## 🧪 测试

```bash
# 全部测试（含真实 AI 调用）
pytest tests/ -v -s

# 跳过 AI 调用，仅跑快速测试
pytest tests/ -v -s -m "not integration"
```

## 🌊 流式测试页面

直接用浏览器打开 `sse_test.html`（无需构建），输入 prompt 后点击"发送"，即可看到打字机效果。

## 🕳️ 踩坑记录

| 现象 | 根因 | 解法 |
|------|------|------|
| `watchfiles` 无限重载 | 日志写入 `logs/` 被监控 | `--reload-exclude logs` |
| `RemoteDisconnected` | 业务码 40001 被当作 HTTP 状态码 | 状态码固定 200，业务码放 body |
| CORS 预检 405 | 未注册 `CORSMiddleware` | 注册并放在**最外层**（最后 add） |
| SSE 不流式（一直缓冲） | `BaseHTTPMiddleware` 缓冲 `StreamingResponse` | 换成**纯 ASGI 中间件** |
| 前端 `data: ` 判断不匹配 | 后端返回 `data:{...}` 无空格 | 前端 `startsWith('data:')` |

## 📄 License

MIT
