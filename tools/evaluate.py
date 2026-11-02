"""
检索层评测：Recall@K / MRR / nDCG@K（二值相关性）。

纯检索，不调生成 LLM；每个 query 只取一次 embedding，所有策略复用。
检索深度 = max(topk)，每个策略只检索一次，指标在 K 上截断。

用法（项目根目录）：
    .venv/Scripts/python.exe tools/evaluate.py
    .venv/Scripts/python.exe tools/evaluate.py --strategies vector,hybrid --topk 3,5,10
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.core.rag import vector_store  # noqa: E402
from app.core.rag.embedding import embed_texts  # noqa: E402
from app.core.rag.retrieval import retrieve_by_strategy  # noqa: E402

STRATEGIES = ("vector", "hybrid", "hybrid_rerank")
CONFIG_KEYS = (
    "RAG_RECALL_K", "RRF_K", "RRF_WEIGHT_VECTOR", "RRF_WEIGHT_BM25",
    "RERANK_TOP_N", "RERANK_MODEL",
    "RAG_TOP_K", "RAG_CHUNK_SIZE", "RAG_CHUNK_OVERLAP",
    "EMBEDDING_MODEL", "EMBEDDING_DIMENSIONS",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        check=True,
    ).stdout


def _git_snapshot() -> dict:
    return {
        "head": _git("rev-parse", "HEAD").decode("utf-8", "replace").strip(),
        "diff_sha256": _sha256(_git("diff", "HEAD")),
        "status_porcelain": _git("status", "--porcelain").decode("utf-8", "replace").splitlines(),
    }


def _config_snapshot(strategy: str) -> dict:
    snap = {key: getattr(settings, key) for key in CONFIG_KEYS}
    snap["RAG_STRATEGY"] = strategy
    snap["corpus_chunk_count"] = vector_store.count()
    return snap


def _query_metrics(returned: list, relevant: set, ks: list) -> dict:
    """returned 已按相关性降序；二值相关性。"""
    metrics = {}
    for k in ks:
        top = returned[:k]
        metrics[f"recall@{k}"] = (
            sum(1 for cid in top if cid in relevant) / len(relevant) if relevant else 0.0
        )
        dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(top, start=1) if cid in relevant)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    # MRR 取整个检索深度内的首个相关项
    first_hit = next((i for i, cid in enumerate(returned, start=1) if cid in relevant), None)
    metrics["mrr"] = 1.0 / first_hit if first_hit else 0.0
    return metrics


def _mean(rows: list) -> dict:
    if not rows:
        return {}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def _rounded(metrics: dict) -> dict:
    return {key: round(value, 4) for key, value in metrics.items()}


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _print_table(title: str, headers: list, rows: list) -> None:
    print(f"\n{title}")
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))


def main() -> None:
    parser = argparse.ArgumentParser(description="检索层评测（Recall@K / MRR / nDCG@K）")
    parser.add_argument("--dataset", default="eval/dataset.json")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--topk", default="3,5,10")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        parser.error(f"未知策略：{unknown}，可选 {STRATEGIES}")

    ks = sorted({int(k) for k in args.topk.split(",") if k.strip()})
    if not ks or min(ks) <= 0:
        parser.error("--topk 必须是正整数")
    depth = max(ks)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    raw = dataset_path.read_bytes()
    dataset = json.loads(raw.decode("utf-8"))
    items = dataset.get("items") or []
    if not items:
        parser.error("数据集为空")

    embed_started = time.perf_counter()
    embeddings = embed_texts([item["question"] for item in items])
    embed_seconds = time.perf_counter() - embed_started

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = {
        "path": str(dataset_path),
        "sha256": _sha256(raw),
        "version": dataset.get("version"),
        "n_items": len(items),
    }
    git_snapshot = _git_snapshot()
    metric_keys = [f"recall@{k}" for k in ks] + ["mrr"] + [f"ndcg@{k}" for k in ks]
    headers = ["strategy"] + [f"Recall@{k}" for k in ks] + ["MRR"] + [f"nDCG@{k}" for k in ks]

    overall_rows = []
    results = []

    for strategy in strategies:
        settings.RAG_STRATEGY = strategy
        started = time.perf_counter()
        per_query = []
        raw_metrics = []
        type_buckets = {}

        for item, q_embed in zip(items, embeddings):
            hits = retrieve_by_strategy(item["question"], top_k=depth, q_embed=q_embed)
            returned = [h["id"] for h in hits]
            relevant = set(item.get("relevant_chunk_ids") or [])
            metrics = _query_metrics(returned, relevant, ks)
            qtype = item.get("type", "unknown")

            per_query.append({
                "id": item["id"],
                "type": qtype,
                "relevant_chunk_ids": sorted(relevant),
                "returned_chunk_ids": returned,
                "metrics": _rounded(metrics),
            })
            raw_metrics.append(metrics)
            type_buckets.setdefault(qtype, []).append(metrics)

        elapsed = time.perf_counter() - started
        overall = _mean(raw_metrics)

        payload = {
            "strategy": strategy,
            "timestamp": timestamp,
            "dataset": dataset_info,
            "depth": depth,
            "topk": ks,
            "metrics": {
                "overall": _rounded(overall),
                "by_type": {t: _rounded(_mean(rows)) for t, rows in type_buckets.items()},
            },
            "config": _config_snapshot(strategy),
            "git": git_snapshot,
            "timings": {
                "embedding_seconds": round(embed_seconds, 3),
                "retrieval_seconds": round(elapsed, 3),
            },
            "per_query": per_query,
        }

        out_path = out_dir / f"{strategy}_{timestamp}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {out_path}")

        results.append(payload)
        overall_rows.append((strategy, *[_fmt(overall[k]) for k in metric_keys]))

    _print_table("总体（10 条冒烟集，D=%d）" % depth, headers, overall_rows)

    type_rows = []
    for qtype in sorted({t for p in results for t in p["metrics"]["by_type"]}):
        for payload in results:
            metrics = payload["metrics"]["by_type"].get(qtype)
            if metrics:
                type_rows.append((qtype, payload["strategy"], *[_fmt(metrics[k]) for k in metric_keys]))
    _print_table("按题型分组", ["type"] + headers, type_rows)


if __name__ == "__main__":
    main()