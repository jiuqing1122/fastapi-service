"""
DashScope 精排（gte-rerank-v2）。

- 同步 requests 调用：retrieve() 跑在 async 端点里，必须设 timeout，
  否则网络一卡就阻塞事件循环，SSE 会一起卡死
- 失败抛 BizException，禁止静默降级为纯向量（会污染实验结论）
"""
import logging
import time

import requests

from app.core.config import settings
from app.core.exceptions import BizException

logger = logging.getLogger(__name__)

RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


def rerank(question: str, hits: list, top_k: int) -> list:
    """对候选 hit 精排，返回按 relevance_score 降序的前 top_k 条。"""
    if not hits:
        return []

    documents = [h["chunk"] for h in hits]
    body = {
        "model": settings.RERANK_MODEL,
        "input": {"query": question, "documents": documents},
        "parameters": {
            "return_documents": False, # 不传原文，只要index + 分数
            "top_n": min(len(documents), settings.RERANK_TOP_N),
        },
    }

    started = time.perf_counter()
    try:
        resp = requests.post(
            RERANK_URL,
            headers={"Authorization": f"Bearer {settings.AliQwen_API_KEY}"},
            json=body,
            timeout=(3, 10),
        )
    except requests.RequestException as e:
        logger.error("精排请求异常 | question=%s | err=%s", question, e, exc_info=True)
        raise BizException(code=50005, message=f"精排请求失败：{e}")

    elapsed_ms = (time.perf_counter() - started) * 1000

    if resp.status_code != 200:
        logger.error(
            "精排 HTTP %s | question=%s | body=%s",
            resp.status_code, question, resp.text[:500],
        )
        raise BizException(code=50005, message=f"精排失败：HTTP {resp.status_code}")

    try:
        results = resp.json()["output"]["results"]
    except (ValueError, KeyError, TypeError) as e:
        logger.error("精排响应解析失败 | question=%s | body=%s", question, resp.text[:500])
        raise BizException(code=50005, message=f"精排响应格式异常：{e}")

    ranked = []
    for r in results:
        idx = r.get("index")
        score = r.get("relevance_score")
        if not isinstance(idx, int) or not 0 <= idx < len(hits) or score is None:
            logger.error("精排返回非法结果 | question=%s | item=%r", question, r)
            raise BizException(code=50005, message=f"精排返回非法结果：{r!r}")
        ranked.append({**hits[idx], "rerank_score": float(score)})

    ranked.sort(key=lambda h: h["rerank_score"], reverse=True)
    logger.info(
        "精排完成 | question=%s | candidates=%d | returned=%d | 耗时=%.0fms",
        question, len(documents), len(ranked), elapsed_ms,
    )
    return ranked[:top_k]