"""
第一步自检脚本：验证「读文件 → 切分 → 向量化 → 写入 Chroma → 检索」完整链路。

用法 （在项目根目录下执行）：
    python test_vector_store.py --file <文件路径>
    python test_vector_store.py --file 员工手册.pdf --question "公司年假有几天" --top-k 3
可选参数：
    --doc-id      测试用的 doc_id，默认 test_doc_001（重复执行会先清掉同 id 旧数据）
    --question    检索问题，默认取第一块内容作为 query
    --top-k       返回条数，默认 3
"""
import argparse
import logging
import os
import sys

# 让脚本能 import app 包（不依赖 cwd）
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

from app.core.rag.loader import load_document
from app.core.rag.splitter import split_pages
from app.core.rag.embedding import embed_texts
from app.core.rag import vector_store

logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("test_vector_store")

def _preview(text: str, n: int = 80) -> str:
    """将长文本截取前 n 个字符生成一行预览，用于日志打印避免刷屏。"""
    # 把换行/回车替换成空格，合并为一行
    one_line = text.replace("\n", " ").replace("\r", " ")
    # 取前 n 个字符，如果原文本超过 n 个则在末尾追加 ...
    return one_line[:n] + ("..." if len(one_line) > n else "")

def main():
    """
    自定义命令行参数解析
    如
    python test_vector_store.py --file 员工手册.pdf --question "公司年假有几天" --top-k 3
    """
    parser = argparse.ArgumentParser(description="向量化管道自检")
    parser.add_argument("--file", required=True, help="待测试的文件路径")
    parser.add_argument("--question", default=None, help="检索问题；默认取第一块内容")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--doc-id", default="test_doc_001")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"[FAIL] 文件不存在：{args.file}")
        sys.exit(1)# 退出程序，返回非 0 状态码表示失败

    filename = os.path.basename(args.file)# 获取文件名，用于后续 collection 名
    abs_path = os.path.abspath(args.file)# 转换为绝对路径，用于后续读取

    #=====1. 加载======
    print(f"\n[1/5] 加载文件：{filename}")
    pages = load_document(abs_path,filename)
    print(f"      加载 {len(pages)} 个原始片段（PDF 为页数，DOCX/TXT 为 1）")
    if not pages:
        print("[FAIL] 未解析出任何文本，检查文件是否为空或扫描版 PDF")
        sys.exit(1)

    #=====2. 切分======
    print(f"\n[2/5] 切分 (chunk_size=800, overlap=100)")
    chunks = split_pages(pages)
    print(f"      切出 {len(chunks)} 块")

    #=====3. 向量化======
    print(f"\n[3/5] 向量化（每批 ≤10 条）")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    print(f"      得到 {len(embeddings)} 个向量，维度 = {len(embeddings[0]) if embeddings else 0}")

    #=====4. 写入======
    print(f"\n[4/5] 写入 Chroma (doc_id={args.doc_id})")
    vector_store.delete_by_doc(args.doc_id)  # 保证脚本可重复执行
    vector_store.add_chunks(args.doc_id, filename, chunks, embeddings)
    print(f"      写入完成，collection 总条数 = {vector_store.count()}")

    #=====5. 检索======
    question = args.question or chunks[0]["text"][:100]
    print(f"\n[5/5] 检索 Top-{args.top_k}")
    print(f"      Query: {_preview(question, 100)}")
    q_emb = embed_texts([question])[0]
    hits = vector_store.query(q_emb, top_k=args.top_k)
    print(f"      命中 {len(hits)} 条:")
    for i,h in enumerate(hits,1):
        m = h["metadata"]
        print(f"\n      [{i}] distance={h['distance']:.4f}  similarity={h['similarity']:.4f}")
        print(f"          doc_id={m.get('doc_id')}  page={m.get('page')}  chunk_index={m.get('chunk_index')}")
        print(f"          preview: {_preview(h['chunk'], 100)}")

        print("\n[DONE] 向量化管道验证通过\n")

if __name__ == "__main__":
    main()