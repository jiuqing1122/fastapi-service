"""
文档加载器：把 PDF/ DOCX / TXT 读成[{"text": ..., "page": ..., "source": ...}, ...]

设计原则:
- 直读，不用 LangChain loader —— pypdf / docx2txt 本身够用，代码更透明
- PDF 保留页码；DOCX 和 TXT 没有"页"的概念，page 统一填 0
- 空页 / 空文件直接跳过，绝不产出空 chunk
pages结构示例：
pages = [
    {"text": "第一章 总则\n第一条 为了规范公司员工考勤管理...", "page": 1, "source": "员工手册.pdf"},
    {"text": "第二条 员工每日工作时间不超过八小时...",       "page": 2, "source": "员工手册.pdf"},
]
"""
import logging
from pypdf import PdfReader
import docx2txt
from app.core.exceptions import BizException

logger = logging.getLogger(__name__)#__name__ 是当前模块的名称（这里是 app.core.rag.loader）

# 支持的文件扩展名
SUPPORTED_EXTS = {"pdf", "docx", "txt"}

def _ext_of(filename: str) -> str:
    """提取文件扩展名，无 '.' 返回空串，不抛异常"""
    if "." not in filename:
        return ""
    return filename.split(".",1)[-1].lower()

def load_pdf(path: str, source: str) -> list:
    """按页读 PDF，页码从 1 开始，单页解析失败不阻断全文"""
    reader = PdfReader(path)#获取PDF文件的所有页面和元数据
    pages = []
    #每次循环返回一个元组，包含当前页面的索引和页面对象，分别对应 i 和 page 变量
    for i,page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""#提取当前页面的文本内容，如果为空则返回空字符串
        except Exception as e:
            logger.warning("PDF 第 %d 页解析失败: %s", i + 1, e)
            text = ""#如果解析失败，也返回空字符串
        text = text.strip()
        if text:
            pages.append({"text": text, "page": i + 1, "source": source})#如果文本内容不为空，就添加到 pages 列表中
    return pages

def load_docx(path: str, source: str) -> list:
    """用 docx2txt 抽全文，DOCX 无固定分页，统一 page=0"""
    try:
        text = docx2txt.process(path) or ""#提取DOCX文件的文本内容，如果为空则返回空字符串
    except Exception as e:
        raise BizException(code=40006,message=f"DOCX 解析失败: {e}")#如果解析失败，抛出异常
    text = text.strip()#移除文本首尾的空格
    if not text:
        return []#如果文本内容为空，就返回空列表
    return [{"text":text,"page":0,"source":source}]

def load_txt(path: str, source: str) -> list:
    """宽松编码探测：utf-8 → utf-8-sig → gbk → latin-1（latin-1 永不失败，兜底）"""
    text = None
    for enc in ("utf-8","utf-8-sig","gbk","latin-1"):
        try:
            with open(path,"r",encoding=enc) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        #理论上不会发生，因为 latin-1 永不失败，但是为了保险起见，还是抛出异常
        raise BizException(code=40005,message="TXT 文件编码无法识别")

    text = text.strip()
    if not text:
        return []
    return [{"text":text,"page":0,"source":source}]

def load_document(path: str, filename: str) -> list:
    """
    统一入口：根据扩展名分派到具体 loader。
    :param path: 文件的绝对路径
    :param filename: 原始文件名（用于取扩展名 + 记录来源）
    :return: [{"text": str, "page": int, "source": str}, ...]
    """
    ext = _ext_of(filename)
    if ext not in SUPPORTED_EXTS:
        raise BizException(code=40004,message=f"不支持的文件类型: {ext or '(无扩展名)'}")

    if ext == "pdf":
        return load_pdf(path, filename)
    if ext == "docx":
        return load_docx(path, filename)
    return load_txt(path, filename)
