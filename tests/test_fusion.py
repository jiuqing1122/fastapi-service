"""RRF 融合纯函数测试：同 id 去重、名次合并、权重倾斜。"""
from app.core.rag.fusion import weighted_rrf


def _hit(chunk_id: str, distance: float | None = 0.1) -> dict:
    return {
        "id": chunk_id,
        "doc_id": "doc",
        "chunk": "text",
        "distance": distance,
        "similarity": None if distance is None else 1.0 - distance,
        "metadata": {},
    }


def _bm25_hit(chunk_id: str, score: float) -> dict:
    hit = _hit(chunk_id, distance=None)
    hit["bm25_score"] = score
    return hit


def test_same_id_dedup_and_rank_merge():
    vector_hits = [_hit("a"), _hit("b")]
    bm25_hits = [_bm25_hit("b", 3.0), _bm25_hit("c", 2.0)]

    out = weighted_rrf(
        vector_hits, bm25_hits, top_k=3,
        k=60, weight_vector=1.0, weight_bm25=1.0,
    )

    assert len(out) == 3  # b 只保留一条
    assert [h["id"] for h in out] == ["b", "a", "c"]
    b = out[0]
    assert b["vector_rank"] == 2 and b["bm25_rank"] == 1
    assert b["distance"] == 0.1       # 保留向量侧字段
    assert b["bm25_score"] == 3.0     # 补上 BM25 原始分
    assert out[2]["distance"] is None and out[2]["similarity"] is None


def test_weight_zero_degrades_to_single_path():
    vector_hits = [_hit("a"), _hit("b"), _hit("c")]
    bm25_hits = [_bm25_hit("c", 9.0), _bm25_hit("a", 8.0)]

    out = weighted_rrf(
        vector_hits, bm25_hits, top_k=3,
        k=60, weight_vector=0.0, weight_bm25=1.0,
    )

    assert [h["id"] for h in out] == ["c", "a", "b"]