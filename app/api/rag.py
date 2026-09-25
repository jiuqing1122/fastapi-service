"""
RAG 相关接口：
- POST   /ai/rag/upload          上传文档
- GET    /ai/rag/documents       文档列表
- GET    /ai/rag/documents/{id}  文档详情
- DELETE /ai/rag/documents/{id}  删除文档
（问答接口在第 3 步加）
"""
import asyncio
import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime

import filetype
from fastapi import APIRouter, UploadFile, File, Query
from starlette.responses import StreamingResponse

from app.core.config import settings
from app.core.exceptions import BizException
from app.core.rag import vector_store, storage, qa
from app.core.rag.embedding import embed_texts
from app.core.rag.loader import load_document
from app.core.rag.splitter import split_pages
from app.models.rag_schemas import RagChatRequest
from app.models.response import APIResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/rag", tags=["RAG"])

#扩展名白名单
ALLOWED_EXTS = {"pdf", "docx", "txt"}

# 各扩展名允许的 MIME（宽松辅助判断，不做硬拒）
ALLOWED_MIMES = {
    "pdf": {"application/pdf"},
    "docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/x-zip-compressed",
    },
    "txt": {"text/plain"},
}

#=======检验辅助=======
def _ext_of(filename: str) -> str:
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".",1)[-1].lower()

def _sanitize_filename(name: str) -> str:
    """
    去路径分隔符、去 ..、去控制字符、截断长度。
    这是路径穿越的第一道防线，落盘前还会有绝对路径复核（第二道）。
    """
    name = os.path.basename(name or "")#取文件名部分，去掉路径
    name = re.sub(r"[\x00-\x1f]", "", name)  # 控制字符
    name = re.sub(r'[\\/:*?"<>|]', "_", name)  # Windows 非法字符 + 路径分隔符
    name = name.replace("..", "_").strip()#把 .. 替换成 _，防止路径穿越攻击，去掉首尾空格
    name = name[:200]  # 截断，避免路径过长
    return name or "unnamed"

def _validate_upload(filename: str, content_type: str, data: bytes) -> str:
    """
    所有上传校验集中在这。通过返回扩展名；失败抛 BizException。
    顺序：扩展名 → 非空 → 大小 → MIME 提示 → magic bytes
    """
    ext = _ext_of(filename)
    if ext not in ALLOWED_EXTS:
        raise BizException(
            code=40001,
            message=f"不支持的文件类型: .{ext or '(无)'}；仅支持 pdf/docx/txt"
        )

    if not data:
        raise BizException(code=40002, message="文件内容为空")

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise BizException(
            code=40003,
            message=f"文件大小不能超过 {settings.MAX_UPLOAD_MB}MB（当前大小 {len(data) / 1024 / 1024:.2f}MB）"
        )

    # MIME 只做提示，不作为硬性拒绝依据（客户端可能不传或乱传）
    ct = (content_type or "").lower().split(";")[0].strip()
    if ct and ct not in ALLOWED_MIMES[ext]:
        logger.warning("MIME 与扩展名不匹配（不阻断）| ext=%s | ct=%s", ext, ct)

    # magic bytes 校验（关键防线）
    head = data[:8192]  # 取文件前 8KB，包含文件格式的魔数标识
    kind = filetype.guess(head)  # 通过魔数猜测文件类型，返回类型对象或 None，如pdf为%PDF

    if ext == "pdf":
        if not kind or kind.mime != "application/pdf":
            raise BizException(code=40004, message="PDF 文件头校验失败，文件可能被篡改或损坏")
    elif ext == "docx":
        # docx 是 zip 容器，magic bytes 是 PK\x03\x04
        if not kind or kind.mime not in (
            "application/zip",
            "application/x-zip-compressed",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ):
            raise BizException(code=40004, message="DOCX 文件头校验失败，文件可能被篡改或损坏")
    elif ext == "txt":
        # 纯文本没有 magic bytes，改用"前 8KB 能解码"判断
        ok = False
        for enc in("utf-8", "gbk", "latin-1"):
            try:
                head.decode(enc)
                ok = True
                break
            except UnicodeDecodeError:
                continue
        if not ok:
            raise BizException(code=40004, message="TXT 文件编码无法识别")
    return ext

#=======上传核心逻辑（同步，跑在线程池里）=======
def _process_upload(filename: str, content_type: str, data: bytes) -> dict:
    """
    校验 → 落盘 → 加载 → 切分 → 向量化 → 存 → 元数据
    任一步失败都回滚：删文件、删向量、删元数据
    """
    ext = _validate_upload(filename, content_type, data)
    doc_id = uuid.uuid4().hex#hex 去除 - 号，生成唯一文档ID
    safe_name = _sanitize_filename(filename)#安全文件名
    doc_dir = os.path.join(settings.UPLOAD_DIR, doc_id)#文档目录
    file_path = os.path.join(doc_dir, safe_name)#文档文件路径

    # 第二道防线：绝对路径复核，确保最终路径在 UPLOAD_DIR 下
    abs_root = os.path.abspath(settings.UPLOAD_DIR)
    abs_final = os.path.abspath(file_path)
    if not abs_final.startswith(abs_root + os.sep):
        raise BizException(code=40005, message="非法文件路径")

    upload_time = datetime.now().isoformat(timespec="seconds")
    saved = False
    try:
        #1.落盘
        os.makedirs(doc_dir, exist_ok=True) #创建目录，如果存在则不创建
        # wb 以二进制写入模式打开文件
        with open(file_path, "wb") as f:
            f.write(data)
            saved = True

        # 2. 加载 → 切分
        pages = load_document(file_path,safe_name)
        if not pages:
            raise BizException(code=40006, message="未从文件中解析出任何文本（可能是扫描版 PDF）")

        chunks = split_pages(pages)
        if not chunks:
            raise BizException(code=40007, message="切分后无有效文本块")

        # 3. 向量化（自动分批）
        embeddings = embed_texts([c["text"] for c in chunks])

        # 4. 写入 Chroma
        vector_store.add_chunks(doc_id, safe_name, chunks, embeddings)

        # 5. 元数据
        # 5. 元数据原子写
        meta = {
            "doc_id": doc_id,
            "filename": safe_name,
            "original_filename": filename,
            "size": len(data),
            "content_type": content_type or "",
            "upload_time": upload_time,
            "chunk_count": len(chunks),
            "page_count": len(pages),
            "ext": ext,
            "status": "ready",
        }
        storage.save_meta(doc_id, meta)
        return meta
    except Exception:
        # 回滚：删文件、删向量、删元数据（都幂等，哪个没做就跳过）
        #删除保存的文件
        if saved:
            shutil.rmtree(doc_dir, ignore_errors=True) #递归删除目录，忽略错误
        try:
            #删除向量
            vector_store.delete_by_doc(doc_id)
        except Exception as e:
            logger.warning("回滚向量失败 | doc_id=%s | err=%s", doc_id, e)
        try:
            storage.delete_meta(doc_id)
        except Exception:
            pass
        raise

#=======路由=======
@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    """上传文档：校验 → 落盘 → 向量化 → 存元数据"""
    raw = await file.read()#读取文件内容
    # 阻塞操作（文件读写、HTTP embedding、Chroma 写）丢线程池，避免卡事件循环
    meta = await asyncio.to_thread(
        _process_upload,
        file.filename or "",
        file.content_type or "",
        raw,
    )
    return APIResponse(code=200, message="success", data=meta)

@router.get("/documents")
async def list_documents(
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        keyword: str = Query(""),
):
    """查询文档列表"""
    items, total = storage.list_meta(page=page, page_size=page_size, keyword=keyword)
    return APIResponse(
        code=200,
        message="success",
        data={"items": items, "total": total, "page": page, "page_size": page_size}
    )

@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """根据ID查询文档"""
    meta = storage.load_meta(doc_id)
    if meta is None:
        raise BizException(code = 40401, message="文档不存在")
    return APIResponse(code=200, message="success", data=meta)

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    删除文档：向量 + 原始文件 + 元数据。
    幂等：不存在时也返回成功（业务上重试删除不报错更友好）。
    """
    def _do_delete() -> bool:
        try:
            #删除向量
            vector_store.delete_by_doc(doc_id)
        except Exception as e:
            logger.warning("删除向量失败（忽略）| doc_id=%s | err=%s", doc_id, e)

        doc_dir = os.path.join(settings.UPLOAD_DIR, doc_id)
        #删除原始文件
        if os.path.isdir(doc_dir):
            shutil.rmtree(doc_dir, ignore_errors=True) #递归删除目录，忽略错误

        #删除元数据
        return storage.delete_meta(doc_id)

    existed = await asyncio.to_thread(_do_delete)

    return APIResponse(
        code=200,
        message="success" if existed else "文档不存在（幂等返回）",
        data={"doc_id": doc_id, "existed": existed}
    )

#=======问答接口=======
@router.post("/chat")
async def rag_chat(request: RagChatRequest):
    """
    RAG 非流式问答
    - 问题为空 -> 40001
    - 检索无结果 -> 40002
    - 正常 -> 200 + {answer, sources}
    """
    if not request.question or not request.question.strip():
        raise BizException(code=40001, message="问题不能为空")

    answer, sources = await qa.answer(request.question)
    if not sources:
        raise BizException(code=40002, message="检索无结果")

    return APIResponse(
        code=200,
        message="success",
        data={"answer": answer, "sources": sources},
    )

@router.post("/chat/stream")
async def rag_chat_stream(request: RagChatRequest):
    """
    RAG 流式问答（SSE）。
    事件顺序：sources → message... → done
    - 请求体校验失败：流开始前返回非 200 业务异常 JSON
    - 流中异常：发 event: error 后立即关流，HTTP 状态码保持 200（流已开始不能改状态码）
    """
    # 1. 流开始前校验（此时还能改状态码，走全局异常处理器返回 200 + 业务码）
    if not request.question or not request.question.strip():
        raise BizException(code=40001, message="问题不能为空")

    question = request.question.strip()

    async def event_generator():
        try:
            async for item in qa.answer_stream(question):
                data_json = json.dumps(item["data"], ensure_ascii=False)
                yield f"event: {item["event"]}\ndata: {data_json}\n\n"
        except Exception as e:
            # 保底：answer_stream 内部已 try/except，这里只兜住"发出去之前"的极端异常
            logger.error("RAG SSE 生成异常 | question=%s | err=%s", question, e, exc_info=True)
            err = json.dumps({"type": "error", "code": 500, "message": "服务内部错误"}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"
            yield f"event: done\ndata: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",# 防止 Nginx 类代理缓冲；本地开发无害
        }
    )