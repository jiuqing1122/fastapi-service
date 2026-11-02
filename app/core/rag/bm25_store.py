"""
BM25 词法召回（论文实验）。

- 数据源：vector_store.get_all_chunks()
- 内存索引，不持久化：语料小，重建成本低
- 失效策略：vector_store.count() 变化时自动重建；数量不变的内容变更
  需要显式调用 rebuild()

shortcut: 只比对 count，删一篇+传一篇（总数不变）不会触发重建；
           语料开始动态变化时改为自增 version 或在上传/删除后显式 rebuild()。
"""
import logging

import jieba
from rank_bm25 import BM25Okapi

from app.core.rag import vector_store

logger = logging.getLogger(__name__)

_corpus: list = []              # get_all_chunks() 的原始条目
_bm25: BM25Okapi | None = None
_indexed_count = -1             # -1 表示尚未建索引


def _tokenize(text: str) -> list:
    """中文分词：jieba.lcut 后去掉空白 token"""
    return [w for w in jieba.lcut(text) if w.strip()]


def rebuild() -> None:
    """全量重建索引；脚本或语料变更后显式调用"""
    global _corpus, _bm25, _indexed_count
    _corpus = vector_store.get_all_chunks()
    tokenized = [_tokenize(c["chunk"]) for c in _corpus]
    _bm25 = BM25Okapi(tokenized) if any(tokenized) else None
    _indexed_count = len(_corpus)
    logger.info("BM25 索引重建完成 | chunks=%d", _indexed_count)


def _ensure_index() -> None:
    # count() 是一次轻量查询，用它判断语料是否变化
    if _indexed_count < 0 or vector_store.count() != _indexed_count:
        rebuild()


def query(text: str, top_k: int = 5) -> list:
    """
    返回与 vector_store.query 同构的命中列表，另带 bm25_score。
    distance/similarity 为 None（BM25 没有向量距离）。
    """
    _ensure_index()
    if _bm25 is None:
        return []

    tokens = _tokenize(text)
    if not tokens:
        return []

    scores = _bm25.get_scores(tokens)
    # 只保留正分：0 分表示没有词命中，送进融合只会污染结果
    order = [i for i in scores.argsort()[::-1] if scores[i] > 0][:top_k]

    hits = []
    for i in order:
        c = _corpus[i]
        hits.append({
            "id": c["id"],
            "doc_id": c["doc_id"],
            "chunk": c["chunk"],
            "distance": None,
            "similarity": None,
            "metadata": c["metadata"],
            "bm25_score": float(scores[i]),
        })
    return hits