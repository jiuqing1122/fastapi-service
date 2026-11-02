"""
检索策略分派（论文实验）。

RAG_STRATEGY:
- vector        : 向量召回，行为与改造前完全一致
- hybrid        : 向量 + BM25 双路召回 → 加权 RRF 融合
- hybrid_rerank : hybrid → DashScope gte-rerank 精排

统一返回结构：{"id", "doc_id", "chunk", "distance", "similarity", "metadata", ...}
"""
from app.core.config import settings
from app.core.rag import vector_store
from app.core.rag.embedding import embed_texts


def retrieve_by_strategy(
    question: str,
    top_k: int | None = None,
    q_embed: list | None = None,
) -> list:
    """
    q_embed: 预计算的 query 向量；评测时传入以复用缓存，服务调用不传。
    """
    top_k = top_k or settings.RAG_TOP_K
    strategy = settings.RAG_STRATEGY
    if strategy not in ("vector", "hybrid", "hybrid_rerank"):
        raise ValueError(f"未知的 RAG_STRATEGY：{strategy}")

    if q_embed is None:
        q_embed = embed_texts([question])[0]

    if strategy == "vector":
        return vector_store.query(q_embed, top_k=top_k)

    # hybrid / hybrid_rerank：延迟导入，vector 策略不加载 jieba / BM25
    from app.core.rag.bm25_store import query as bm25_query
    from app.core.rag.fusion import weighted_rrf

    vector_hits = vector_store.query(q_embed, top_k=settings.RAG_RECALL_K)
    bm25_hits = bm25_query(question, top_k=settings.RAG_RECALL_K)

    # hybrid_rerank 先融合到 RERANK_TOP_N 再精排；hybrid 直接截到 top_k
    limit = max(settings.RERANK_TOP_N, top_k) if strategy == "hybrid_rerank" else top_k
    fused = weighted_rrf(
        vector_hits, bm25_hits, limit,
        k=settings.RRF_K,
        weight_vector=settings.RRF_WEIGHT_VECTOR,
        weight_bm25=settings.RRF_WEIGHT_BM25,
    )
    if strategy == "hybrid":
        return fused

    from app.core.rag.rerank import rerank
    return rerank(question, fused, top_k=top_k)