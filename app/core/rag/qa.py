"""
RAG 问答核心逻辑：
- retrieve()     ：问题 → 向量化 → Chroma 检索 → 命中列表
- build_prompt() ：命中列表 + 原问题 → 拼成给 LLM 的 prompt
- answer()       ：非流式问答，返回 (answer, sources)
- answer_stream()：流式问答，async generator 产出 SSE 事件字典
"""
import logging

from app.core.ai_client import chat_with_ai, chat_with_ai_stream
from app.core.config import settings
from app.core.rag import vector_store
from app.core.rag.embedding import embed_texts


logger = logging.getLogger(__name__)

#======检索======
def retrieve(question: str, top_k: int | None = None) -> list:
    """
    问题向量化 + Chroma 检索。
    返回：[{"doc_id","chunk","distance","similarity","metadata"}, ...]
    """
    top_k = top_k or settings.RAG_TOP_K
    q_embed = embed_texts([question])
    return vector_store.query(q_embed[0], top_k=top_k)

#======拼 prompt======
_SYSTEM_PROMPT = """你是一个严谨的知识库助手。请严格遵守以下规则：

1. 只根据下面提供的【参考资料】回答用户问题，不要使用参考资料之外的知识。
2. 如果参考资料中没有相关信息，直接回答"根据现有资料无法回答该问题"，不要编造。
3. 回答时用中文，简洁准确。如引用了具体片段，可以说明"根据片段 N"。
4. 不要提及"参考资料"这个词本身，直接给答案即可。"""

def build_prompt(question: str, hits: list) -> str:
    """
    把命中的片段拼成带编号的参考资料，加上原始问题。
    """
    if not hits:
        return f"{_SYSTEM_PROMPT}\n\n用户问题：\n{question}"

    blocks = []
    for i,h in enumerate(hits, 1):
        meta = h.get("metadata") or {}
        page = meta.get("page",0)
        source = meta.get("source","")
        # 页码为 0 说明是 DOCX/TXT（无分页概念），不展示页号
        page_str = f"第{page}页" if page else ""
        blocks.append(f"【片段 {i}】（来源：{source}{page_str}）\n{h['chunk']}")

    refs = "\n\n".join(blocks)
    return f"{_SYSTEM_PROMPT}\n\n【参考资料】\n{refs}\n\n【用户问题】\n{question}"

def _build_sources(hits: list) -> list:
    """
    构造给前端展示的 sources 列表。
    chunk 字段仅保留前 200 字，避免响应体过大。
    """
    sources = []
    for h in hits:
        meta = h.get("metadata") or {}
        chunk = h.get("chunk")
        sources.append({
            "doc_id": h.get("doc_id"),
            "chunk": chunk[:200] + ("..." if len(chunk) > 200 else ""),
            "distance": h.get("distance"),
            "similarity": h.get("similarity"),
            "metadata": meta,
        })
    return sources

async def answer(question: str) -> tuple[str, list]:
    """
    非流式 RAG 问答。
    返回 (answer_text, sources)。
    检索无结果时返回 ("", [])，由调用方决定怎么响应。
    """
    hits = retrieve(question)
    if not hits:
        return "", []

    sources = _build_sources(hits)
    prompt = build_prompt(question, hits)
    reply = await chat_with_ai(prompt)
    return reply, sources

async def answer_stream(question: str):
    """
    流式 RAG 问答。
    yield 字典，每个字典对应一个 SSE 事件：
        {"event": "source", "data": {...}}
        {"event": "message", "data": {...}}
        {"event": "done",    "data": {...}}
        {"event": "error",   "data": {...}}
    """
    # 检索
    try:
        hits = retrieve(question)
    except Exception as e:
        logger.error("RAG 检索失败 | question=%s | err=%s", question, e, exc_info=True)
        yield {"event": "error", "data": {"type": "error", "code": 50001, "message": f"检索失败：{e}"}}
        return

    # 无命中：发空 sources + done，不调 LLM（没证据，调了也是编）
    if not hits:
        yield {"event": "sources", "data": {"type": "sources", "sources": []}}
        yield {"event": "done", "data": {"type": "done"}}
        return

    # 有命中：先发 sources, 再流式生成 message
    sources = _build_sources(hits)
    yield {"event": "sources", "data": {"type": "sources", "sources": sources}}

    prompt = build_prompt(question, hits)
    try:
        async for delta in chat_with_ai_stream(prompt):
            yield {"event": "message", "data": {"type": "message", "content": delta}}
    except Exception as e:
        logger.error("RAG 流式生成失败 | question=%s | err=%s", question, e, exc_info=True)
        yield {"event": "error", "data": {"type": "error", "code": 50002, "message": f"生成失败：{e}"}}
        return

    yield {"event": "done", "data": {"type": "done"}}
