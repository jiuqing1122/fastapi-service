"""
RAG 相关的 Pydantic 模型。

- DocumentMeta / DocumentListResponse：文档元数据
- RagChatRequest：RAG 聊天请求体
"""
from typing import List

from pydantic import BaseModel

class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    original_filename: str = ""# 原始文件名
    size: int# 文件大小，单位字节
    content_type: str = ""
    upload_time: str
    chunk_count: int
    page_count: int
    ext: str# 文件扩展名，如 .pdf、.docx 等
    status: str = "ready"

class DocumentListResponse(BaseModel):
    items: list[DocumentMeta]
    total: int
    page: int
    page_size: int

class RagChatRequest(BaseModel):
    """RAG 问答请求体"""
    question: str