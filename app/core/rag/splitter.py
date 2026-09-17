"""
文本切分器：把 loader 输出的段落切成语义相对完整的 chunk。

为什么用 RecursiveCharacterTextSplitter：
- 它按 separators 列表从左到右尝试，先保大段、再退到小段，符合"先段后句"的直觉
- 自带 chunk_overlap，跨块边界不会丢上下文
- 默认 separators 缺中文标点，这里手动补上
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings

# 中文友好的分隔符：段落 → 换行 → 中文句末标点 → 英文句末标点 → 空白 → 单字符
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]

# 文本切分器实例
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.RAG_CHUNK_SIZE,
    chunk_overlap=settings.RAG_CHUNK_OVERLAP,
    length_function=len,
    separators=_SEPARATORS,
)

def split_pages(pages: list) -> list:
    """
    对每个 page 单独切分，保留 page 信息。
    返回：[{"text": str, "page": int, "chunk_index": int, "source": str}, ...]

    chunk_index 在整个文档范围内全局递增，用于构造稳定的 Chroma id。
    为什么按页单独切：一个 chunk 如果跨页，它的 page 归属就说不清；
    按页切保证每个 chunk 天然属于某一页，检索结果展示来源时不会含糊。
chunks结构示例：
    chunks = [
        {"text": "第一章 总则 第一条 为了规范公司员工考勤管理", "page": 1, "chunk_index": 0, "source": "员工手册.pdf"},
        {"text": "第二条 员工每日工作时间不超过八小时",         "page": 2, "chunk_index": 1, "source": "员工手册.pdf"},
        {"text": "第三十条 本办法自发布之日起执行",             "page": 5, "chunk_index": 2, "source": "员工手册.pdf"},
    ]
    """
    chunks = []
    idx = 0
    # 遍历每一页文本
    for p in pages:
        # 对当前页文本进行切分，遍历得到的小块
        for piece in _splitter.split_text(p["text"]):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append({
                    "text": piece,
                    "page": p["page"],
                    "chunk_index": idx,
                    "source": p.get("source", ""),# 如果当前页有 source 信息，就用它；没有就用空字符串代替
                })
            idx += 1
    return chunks