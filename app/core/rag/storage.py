"""
文档元数据存储：JSON 文件 + 原子写。

存储位置：data/docs/{doc_id}.json
原子写：先写 {doc_id}.json.tmp → flush → os.replace 覆盖
这样进程被杀也不会留下半截 JSON
"""
import json
import logging
import os
from app.core.config import settings

logger = logging.getLogger(__name__)

def _meta_path(doc_id: str) -> str:
    return os.path.join(settings.DOC_META_DIR,f"{doc_id}.json")

def save_meta(doc_id: str, meta: dict) -> None:
    """原子写：先写 .tmp 再 replace"""
    path = _meta_path(doc_id)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.flush() #把 Python 缓冲区的数据推到操作系统
        os.fsync(f.fileno()) # 强制操作系统把数据写入磁盘，不只是内存
    os.replace(tmp, path) # 原子替换，确保文件内容完整

def load_meta(doc_id: str) -> dict | None:
    path = _meta_path(doc_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("读取元数据失败 | doc_id=%s | err=%s",doc_id, e)
        return None

def delete_meta(doc_id: str) -> bool:
    """返回是否真的删除了文件（幂等：不存在时返回 False）"""
    path = _meta_path(doc_id)
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False

def list_meta(page: int = 1, page_size: int = 10, keyword: str = "") -> tuple[list, int]:
    """
    返回 (items, total)。items 按 upload_time 倒序，keyword 模糊匹配 filename。
    keyword 为空时不做过滤。
    """
    if not os.path.isdir(settings.DOC_META_DIR):
        return [], 0

    all_metas = []
    for name in os.listdir(settings.DOC_META_DIR):
        if not name.endswith(".json") or name.endswith(".json.tmp"):
            continue
        try:
            with open(os.path.join(settings.DOC_META_DIR, name), "r", encoding="utf-8") as f:
                all_metas.append(json.load(f))
        except Exception as e:
            # 单个文件坏了不影响整体列表，跳过即可
            logger.warning("跳过损坏的元数据文件 | file=%s | err=%s", name, e)

    if keyword:
        kw = keyword.lower()
        all_metas = [m for m in all_metas if kw in (m.get("filename") or "").lower()]

    all_metas.sort(key=lambda m: m.get("upload_time", ""), reverse=True)

    total = len(all_metas)
    start = (page - 1) * page_size
    end = start + page_size
    return all_metas[start:end], total