from fastapi import FastAPI
from app.api import chat, rag
from app.core.logging_config import setup_logging
from app.core.exception_handlers import biz_exception_handler,global_exception_handler
from app.core.exceptions import BizException
from app.core.middleware import LoggingMiddleware
from fastapi.middleware.cors import CORSMiddleware

# 初始化日志
setup_logging()

app = FastAPI(
    title="AI Microservice",
    version="1.0",
    description="一个简单的 AI 网关服务，集成 DeepSeek"
)

# 注册中间件（先于异常处理器，但异常最终会进入处理器）
app.add_middleware(LoggingMiddleware)

# 注册 CORS 中间件，解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 开发阶段允许所有源；生产应改为具体域名
    allow_credentials=False,       # 与 allow_origins=["*"] 配合必须为 False
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册全局异常处理器
app.add_exception_handler(BizException, biz_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

# 注册路由
app.include_router(chat.router)
app.include_router(rag.router)

# 根路由，返回服务状态
@app.get("/")
async def root():
    return {"message": "AI Service is running"}

#方便PyCharm直接运行
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
