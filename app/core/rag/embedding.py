"""
Embedding 封装：调阿里云百炼 DashScope 的 OpenAI 兼容接口。

三条硬约束：
1. DashScope text-embedding-v3 单次请求最多 10 条 —— 硬限制，SDK 不会帮你切
2. 返回的向量顺序必须和传入文本顺序严格一致，否则向量与文本错位
3. 失败要抛 BizException，不能静默跳过 —— 否则出现"上传成功但检索不到"
"""
import logging

from openai import OpenAI

from app.core.config import settings
from app.core.exceptions import BizException

logger = logging.getLogger(__name__)
# 懒加载单例，避免 import 时就建连接
_client = None

def _get_client() -> OpenAI:
    global _client#声明函数内部要修改模块级别的全局变量 _client，而不是创建局部变量
    if not _client:
        _client = OpenAI(
            api_key=settings.AliQwen_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL
        )
    return _client

def embed_texts(texts: list) -> list:
    """
    批量向量化，自动分批（每批 ≤ settings.EMBEDDING_BATCH_SIZE，DashScope 上限 10）。
    :param texts: List[str]
    :return: List[List[float]]，顺序与入参一一对应
    embedding结构示例：
    embeddings = [
    [0.0123, -0.0456, 0.0789, ..., 0.0012, 0.0567],   # chunks[0] 的向量（1024 维）
    [0.0345,  0.0678, -0.0123, ..., 0.0890, -0.0234],  # chunks[1] 的向量
    [0.0567, -0.0890,  0.0345, ..., -0.0456, 0.0123],  # chunks[2] 的向量
    ]
    """
    if not texts:
        return []

    client = _get_client()
    batch_size = settings.EMBEDDING_BATCH_SIZE
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            # 调用 DashScope API 向量化
            resp = client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=batch,
                dimensions=settings.EMBEDDING_DIMENSIONS,
            )
        except Exception as e:
            logger.error("Embedding 调用失败 | batch_start=%d | size=%d | err=%s",
                         start, len(batch), e, exc_info=True)
            raise BizException(
                code=50003,
                message=f"Embedding 调用失败（批次起点 {start}，{len(batch)} 条）：{e}",
            )

        # 按 index 排序，确保向量和入参文本顺序严格一致
        # resp.data = [
        #     EmbeddingObject(embedding=[0.0123, -0.0456, ...], index=5),
        #     EmbeddingObject(embedding=[0.0345, 0.0678, ...], index=0),
        #     EmbeddingObject(embedding=[0.0567, -0.0890, ...], index=3),
        # ]
        # items = [
        #     EmbeddingObject(embedding=[0.0345, 0.0678, ...], index=0),  # 对应第1个文本
        #     EmbeddingObject(embedding=[...], index=1),  # 对应第2个文本
        #     EmbeddingObject(embedding=[0.0567, -0.0890, ...], index=3),  # 对应第4个文本
        #     EmbeddingObject(embedding=[...], index=4),  # 对应第5个文本
        #     EmbeddingObject(embedding=[0.0123, -0.0456, ...], index=5),  # 对应第6个文本
        # ]
        items = sorted(resp.data, key=lambda x: x.index)
        if len(items) != len(batch):
            raise BizException(
                code=50004,
                message=f"Embedding 返回条数不匹配：请求 {len(batch)} 条，返回 {len(items)} 条",
            )
        #生成器表达式（Generator Expression），遍历 items 里的每个 EmbeddingObject，只取出它的 .embedding 属性
        # all_embeddings = [
        #     [0.0345, 0.0678, ...],  # texts[0] 的向量
        #     [0.0789, -0.0123, ...],  # texts[1] 的向量
        #     ...
        #     [0.0567, -0.0890, ...],  # texts[24] 的向量
        # ]
        all_embeddings.extend(item.embedding for item in items)
    return all_embeddings
