"""
纯检索脚本：跳过加载/切分/写入，直接对已有向量库做查询。

用法：
    python query_only.py --question "把这个文件里最重要的一句话找出来" --top-k 5
"""
import argparse
import sys
import os

# 让脚本能 import app 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.rag.embedding import embed_texts
from app.core.rag import vector_store


def _preview(text: str, n: int = 80) -> str:
    """将长文本截取前 n 个字符生成一行预览。"""
    one_line = text.replace("\n", " ").replace("\r", " ")
    return one_line[:n] + ("..." if len(one_line) > n else "")


def main():
    parser = argparse.ArgumentParser(description="向量库纯检索")
    parser.add_argument("--question", required=True, help="检索问题")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print(f"\n[检索] Query: {_preview(args.question, 100)}")
    print(f"       向量库总条数: {vector_store.count()}")

    q_emb = embed_texts([args.question])[0]
    hits = vector_store.query(q_emb, top_k=args.top_k)

    print(f"       命中 {len(hits)} 条:")
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        print(f"\n       [{i}] distance={h['distance']:.4f}  similarity={h['similarity']:.4f}")
        print(f"           doc_id={m.get('doc_id')}  page={m.get('page')}  chunk_index={m.get('chunk_index')}")
        print(f"           preview: {_preview(h['chunk'], 100)}")

    print("\n[DONE] 检索完成\n")


if __name__ == "__main__":
    main()