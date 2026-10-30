"""
全文搜索 Chroma chunk，供人工标注找候选。

只负责找候选，不判断相关性；最终 relevant_chunk_ids 仍由你确认。

用法（项目根目录）：
    .venv/Scripts/python.exe tools/search_chunks.py --keyword 时效
    .venv/Scripts/python.exe tools/search_chunks.py --keyword 时效 --limit 10
    .venv/Scripts/python.exe tools/search_chunks.py --id 2a1ca15f03db4c54806d572f7eb44a28_3

结果写到 UTF-8 文件（默认 D:/cc-rubbish/search_results.txt），用 VS Code 打开看。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.rag import vector_store  # noqa: E402


def _context(text: str, keyword: str, width: int = 200) -> str:
    """截取关键词前后的上下文，避免整块刷屏。"""
    pos = text.find(keyword)
    if pos < 0:
        return text[:width * 2]
    start = max(0, pos - width)
    end = min(len(text), pos + len(keyword) + width)
    return text[start:end]


def main() -> None:
    parser = argparse.ArgumentParser(description="全文搜索 chunk（只找候选，不判相关性）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--keyword", help="要搜索的关键词")
    group.add_argument("--id", help="按 chunk id 输出完整文本")
    parser.add_argument("--limit", type=int, default=20, help="最多显示多少条，默认 20")
    parser.add_argument("--out", default=r"D:/cc-rubbish/search_results.txt", help="输出文件")
    args = parser.parse_args()

    chunks = vector_store.get_all_chunks()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.id:
        rows = [c for c in chunks if c["id"] == args.id]
        if not rows:
            print("not found:", args.id)
            return
        c = rows[0]
        meta = c.get("metadata") or {}
        with out.open("w", encoding="utf-8") as f:
            f.write(f"{c['id']} | {meta.get('source', '')} | page {meta.get('page', 0)}\n\n")
            f.write(c["chunk"])
        print(f"written: {out}")
        return

    all_matches = [c for c in chunks if args.keyword in c["chunk"]]
    matches = all_matches[:args.limit]
    with out.open("w", encoding="utf-8") as f:
        f.write(f"keyword: {args.keyword}\n")
        f.write(f"matches: {len(all_matches)} (shown {len(matches)})\n\n")
        for c in matches:
            meta = c.get("metadata") or {}
            f.write(f"{c['id']} | {meta.get('source', '')} | page {meta.get('page', 0)}\n")
            f.write(_context(c["chunk"], args.keyword) + "\n")
            f.write("-" * 60 + "\n")
    print(f"matches: {len(all_matches)} (shown {len(matches)}) -> {out}")


if __name__ == "__main__":
    main()