# 项目结构

```
fastapi-ai-service/
├── app/
│   ├── main.py                      # 应用入口：CORS + 全局异常处理器 + ASGI 日志中间件
│   ├── api/
│   │   └── chat.py                  # /ai/chat（非流式）与 /ai/chat/stream（SSE 流式）已跑通
│   ├── core/
│   │   ├── config.py                # 支持 DeepSeek / Ollama 切换，从 .env 读取 Key
│   │   ├── ai_client.py             # AsyncOpenAI 封装：chat_with_ai + chat_with_ai_stream
│   │   ├── exceptions.py            # BizException（业务异常，含 code + message）
│   │   ├── exception_handlers.py    # 全局异常处理：BizException 返回 200+code，其余 500
│   │   ├── logging_config.py        # 控制台 + 滚动文件日志
│   │   └── middleware.py            # 纯 ASGI 日志中间件（不缓冲 StreamingResponse）
│   └── models/
│       ├── schemas.py               # ChatRequest / ChatResponse
│       └── response.py              # 统一响应 APIResponse{code,message,data}
├── tests/                           # pytest，覆盖 7 种场景                    # 流式测试页（前端）
├── logs/app.log
├── .env      # DEEPSEEK_API_KEY / OLLAMA_BASE_URL / AI_SERVICE_TYPE
├──PLAN.md   # 项目文档
└──requirements.txt

```

# 项目启动

## 方式一：使用启动脚本
```bash
python run.py
```

## 方式二：直接运行 main.py（适合 PyCharm）
```bash
python app/main.py
```

## 方式三：使用 uvicorn 命令
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
启动成功后，服务运行在：`http://localhost:8000`

# 运行测试

## 运行全部测试（含真实 AI 调用）
```bash
pytest tests/ -v -s
```

## 只跑快速测试（跳过 integration 标记）
```bash
pytest tests/ -v -s -m "not integration"
```

# 接口文档地址

- **Swagger UI**：http://localhost:8000/docs
- **ReDoc**：http://localhost:8000/redoc