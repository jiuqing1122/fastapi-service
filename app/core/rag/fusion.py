"""
加权 RRF 融合（论文实验）。

score(d) = Σ_i w_i / (k + rank_i(d))，rank 从 1 开始。
纯函数：k 与权重全部显式传入，配置由 retrieval.py 读取。

融合结果每条带：
- rrf_score     融合分
- vector_rank   向量路名次（不在该路则为 None）
- bm25_rank     BM25 路名次（不在该路则为 None）
- bm25_score    BM25 原始分（仅 BM25 命中时有值）
- distance/similarity 保留向量侧的值；BM25 独有来源为 None
"""


def weighted_rrf(
    vector_hits: list,
    bm25_hits: list,
    top_k: int,
    *,
    k: int,
    weight_vector: float,
    weight_bm25: float,
) -> list:
    merged = {}

    for rank, hit in enumerate(vector_hits, start=1):
        merged[hit["id"]] = {
            **hit,
            "bm25_score": None,
            "rrf_score": weight_vector / (k + rank),
            "vector_rank": rank,
            "bm25_rank": None,
        }

    for rank, hit in enumerate(bm25_hits, start=1):
        entry = merged.get(hit["id"])
        if entry is None:
            merged[hit["id"]] = {
                **hit,
                "rrf_score": weight_bm25 / (k + rank),
                "vector_rank": None,
                "bm25_rank": rank,
            }
        else:
            # 两路都命中：分数累加，保留向量侧字段
            entry["rrf_score"] += weight_bm25 / (k + rank)
            entry["bm25_rank"] = rank
            entry["bm25_score"] = hit.get("bm25_score")

    return sorted(merged.values(), key=lambda h: h["rrf_score"], reverse=True)[:top_k]