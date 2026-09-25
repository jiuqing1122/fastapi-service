import os
from dotenv import load_dotenv

load_dotenv()  # 从.env文件中载入环境变量

# 项目根目录：config.py 位于 app/core/ 下，往上三级即 fastapi-service 根
# 用 __file__ 推导，避免 PyCharm 直跑与命令行 uvicorn 工作目录不同
# 导致 data/ 落到两个地方（表现为「上传成功但检索不到」）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _resolve_data_path(env_key: str, default_relative: str) -> str:
    """把 .env 里的相对路径按项目根目录解析为绝对路径；若已是绝对路径则原样返回"""
    raw = os.getenv(env_key, default_relative).strip() or default_relative
    if os.path.isabs(raw):
        return raw
    return os.path.abspath(os.path.join(BASE_DIR, raw))

# 定义配置类
class Settings:
    # DeepSeek 配置
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    
    # Ollama 配置
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

    #qwen配置
    QWEN_BASE_URL: str = os.getenv("QWEN_BASE_URL", "https://ws-sqyuqqrcd3xtnu3r.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
    QWEN_MODEL: str = os.getenv("QWEN_MODEL", "qwen3.8-flash")
    AliQwen_API_KEY: str = os.getenv("AliQwen_API_KEY", "")

    # AI 服务类型选择：deepseek 或 ollama, qwen
    AI_SERVICE_TYPE: str = os.getenv("AI_SERVICE_TYPE", "qwen")
    
    # 通用模型配置
    AI_MODEL: str = os.getenv("AI_MODEL", "deepseek-chat")

    # Embedding 配置
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
    # DashScope 硬限制：单批 ≤10 条，SDK 不会帮你切
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))

    # RAG 配置
    CHROMA_DIR: str = _resolve_data_path("CHROMA_DIR", "data/chroma")
    UPLOAD_DIR: str = _resolve_data_path("UPLOAD_DIR", "data/uploads")
    DOC_META_DIR: str = _resolve_data_path("DOC_META_DIR", "data/docs")
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))# RAG 返回的文档数量
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))# 文档分块大小
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))# 文档分块重叠大小
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "20"))# 最大上传文件大小（MB）

    def validate(self) ->None:
        """
        启动期校验配置；不合法直接抛错，避免运行期报错。
        注意：判断空值必须用 .strip()，因为 .env 里 `KEY=` 返回的是空字符串不是 None。
        """
        if self.AI_SERVICE_TYPE == "qwen" and not self.AliQwen_API_KEY.strip():
            raise ValueError("使用 qwen 时，AliQwen_API_KEY 必须在环境变量中")
        if self.AI_SERVICE_TYPE == "deepseek" and not self.DEEPSEEK_API_KEY.strip():
            raise ValueError("AI_SERVICE_TYPE=deepseek 时，DEEPSEEK_API_KEY 必须配置")
        if not self.EMBEDDING_BASE_URL.strip():
            raise ValueError("EMBEDDING_BASE_URL 必须配置")
        if self.EMBEDDING_BATCH_SIZE > 10:
            raise ValueError("EMBEDDING_BATCH_SIZE 不能超过 10（DashScope 硬限制）")
            # 数据目录自动创建
        for path in (self.CHROMA_DIR, self.UPLOAD_DIR, self.DOC_META_DIR):
            os.makedirs(path, exist_ok=True)

# 创建配置实例
settings = Settings()
#启动时校验配置
settings.validate()
