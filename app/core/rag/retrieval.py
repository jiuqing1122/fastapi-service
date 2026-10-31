"""
检索策略分派（论文实验）。

RAG_STRATEGY:
- vector        : 向量召回，行为与改造前完全一致
- hybrid        : 向量 + BM25 双路召回 → 加权 RRF 融合（第 3、4 步接入）
- hybrid_rerank : hybrid → 精排（第 5 步接入）

统一返回结构：{"id", "doc_id", "chunk", "distance", "similarity", "metadata", ...}
"""
from app.core.config import settings
from app.core.rag import vector_store
from app.core.rag.embedding import embed_texts


def retrieve_by_strategy(question: str, top_k: int | None = None) -> list:
    top_k = top_k or settings.RAG_TOP_K
    strategy = settings.RAG_STRATEGY

    if strategy == "vector":
        q_embed = embed_texts([question])
        return vector_store.query(q_embed[0], top_k=top_k)

    if strategy == "hybrid":
        raise NotImplementedError("hybrid 策略将在第 3、4 步接入")

    if strategy == "hybrid_rerank":
        raise NotImplementedError("hybrid_rerank 策略将在第 3~5 步接入")

    raise ValueError(f"未知的 RAG_STRATEGY：{strategy}")