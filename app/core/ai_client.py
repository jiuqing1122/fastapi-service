from openai import AsyncOpenAI
from app.core.config import settings

# 根据配置初始化对应的 AI 客户端
if settings.AI_SERVICE_TYPE == "ollama":
    # Ollama 也兼容 OpenAI API 格式，使用 Ollama 的 base_url
    client = AsyncOpenAI(
        api_key="ollama",  # Ollama 不需要 API Key，但 openai 库要求必须有值
        base_url=settings.OLLAMA_BASE_URL
    )
    current_model = settings.OLLAMA_MODEL
elif settings.AI_SERVICE_TYPE == "qwen":
    client = AsyncOpenAI(
        api_key=settings.AliQwen_API_KEY,
        base_url=settings.QWEN_BASE_URL
    )
    current_model = settings.QWEN_MODEL
else:
    # 默认使用 DeepSeek
    client = AsyncOpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL
    )
    current_model = settings.AI_MODEL


async def chat_with_ai(prompt: str) -> str:
    """
    调用 AI 非流式聊天补全接口（支持 DeepSeek 和 Ollama）
    :param prompt: 用户输入
    :return: AI 回复内容
    """
    response = await client.chat.completions.create(
        model = current_model,
        messages = [{"role": "user", "content": prompt}],
        stream = False, # 非流式响应
    )
    # 提取回复内容
    return response.choices[0].message.content

async def chat_with_ai_stream(prompt: str):
    """
    调用 AI 流式聊天补全接口（支持 DeepSeek 和 Ollama）
    :param prompt: 用户输入
    :yield: 逐步返回的 AI 回复文本片段 (delta)
    """
    # 调用 OpenAI 兼容接口，关键参数 stream=True
    stream = await client.chat.completions.create(
        model=current_model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,  # 开启流式
    )

    # 异步迭代响应块
    async for chunk in stream:
        # 先检查 choices 是否为空，防止索引越界
        if not chunk.choices:
            continue
        
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta