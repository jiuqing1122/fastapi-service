"""
导出 Chroma 中的全部 chunk，供人工标注评测集。

用法（项目根目录）：
    .venv/Scripts/python.exe tools/list_chunks.py --out eval/chunks.json
    .venv/Scripts/python.exe tools/list_chunks.py --out eval/chunks.json --doc-id <doc_id>
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.rag import vector_store  # noqa: E402


def _preview(text: str, n: int = 120) -> str:
    """压成一行，截前 n 个字符。"""
    one_line = " ".join((text or "").split())
    return one_line[:n] + ("..." if len(one_line) > n else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 Chroma chunk 供标注")
    parser.add_argument("--out", default="eval/chunks.json", help="输出 JSON 路径")
    parser.add_argument("--doc-id", default=None, help="只导出某个文档")
    parser.add_argument("--preview-chars", type=int, default=120, help="预览字符数，默认 120")
    args = parser.parse_args()

    chunks = vector_store.get_all_chunks()
    if args.doc_id:
        chunks = [c for c in chunks if c.get("doc_id") == args.doc_id]

    def sort_key(c: dict) -> tuple:
        meta = c.get("metadata") or {}
        return (
            meta.get("source") or "",
            int(meta.get("page") or 0),
            int(meta.get("chunk_index") or 0),
        )

    chunks.sort(key=sort_key)

    items = []
    for c in chunks:
        meta = c.get("metadata") or {}
        items.append({
            "id": c.get("id"),
            "doc_id": c.get("doc_id"),
            "filename": meta.get("source") or "",
            "page": int(meta.get("page") or 0),
            "preview": _preview(c.get("chunk") or "", args.preview_chars),
            "text": c.get("chunk") or "",
        })

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"total": len(items), "items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total = vector_store.count()
    print(f"exported: {len(items)} chunks -> {out_path}")
    print(f"collection count: {total}")
    if not args.doc_id and len(items) != total:
        print("WARN: exported count != collection count", file=sys.stderr)


if __name__ == "__main__":
    main()