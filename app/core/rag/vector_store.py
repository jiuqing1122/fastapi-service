"""
Chroma 向量库封装。

约定：
- collection 名固定 rag_documents
- 距离空间：cosine（创建时显式指定 hnsw:space = "cosine"）
  - Chroma 默认是 L2，不指定会拿到 L2 距离，不能直接叫"相似度"
  - cosine 空间下：distance = 1 - cosine_similarity，所以 similarity = 1 - distance
- chunk id: f"{doc_id}_{chunk_index}"，便于按 doc 删除
- 单 worker：本地 PersistentClient 在多 worker 下可能锁冲突
"""
import logging
import chromadb
from app.core.config import settings

logger = logging.getLogger(__name__)
COLLECTION_NAME = "rag_documents"

_client = None
_collection = None

def _get_collection():
    global _client, _collection
    # 懒加载：首次调用才创建客户端和集合，避免 import 时连接磁盘
    if _collection is None:
        # 初始化 Chroma 客户端，指定数据目录
        _client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        # 获取已有集合 / 不存在则新建；显式指定 cosine 距离（Chroma 默认 L2）
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},# 显式指定 cosine 距离空间
        )
    return _collection

def add_chunks(doc_id: str, filename: str, chunks: list, embeddings: list) -> None:
    """
    :param doc_id: 文档 ID，用于唯一标识文档
    :param filename: 文档文件名，用于记录来源文件
    :param chunks:     [{"text", "page", "chunk_index", "source"}, ...]
    :param embeddings: List[List[float]]，与 chunks 一一对应
    """
    if not chunks:
        return
    if len(chunks) != len(embeddings):
        raise ValueError(f"chunks({len(chunks)}) 与 embeddings({len(embeddings)}) 数量不一致")

    # 获取集合
    collection = _get_collection()
    ids = [f"{doc_id}_{c['chunk_index']}" for c in chunks]
    documents = [c['text'] for c in chunks]
    # Chroma 的 metadata 值只接受 str / int / float / bool，page 可能是 None，统一转 int
    metadata = [
        {
            "doc_id": doc_id,
            "source": filename,
            "page": int(c.get("page") or 0),
            "chunk_index": int(c['chunk_index']),
        }
        for c in chunks
    ]
    # 插入数据
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadata,
    )

def query(question_embedding: list, top_k: int = 5) -> list:
    """"
    返回：[{"doc_id", "chunk", "distance", "similarity", "metadata"}, ...]
    无命中时返回[]
    """
    collection = _get_collection()
    result = collection.query(
        query_embeddings = [question_embedding],# 查询向量，必须是列表
        n_results = top_k,# 返回 top_k 个结果
    )

    hits = []
    ids = result.get("ids") or [[]]
    if not ids or not ids[0]:
        return hits

    # Chroma 返回的是"每个 query 一组"，我们只传了一个 query，取第 0 组
    #result的大致结构如下
    # result = {
    #     "ids": [["a_0", "a_1", "b_0"]],
    #     "distances": [[0.12, 0.25, 0.33]],
    #     "documents": [["RAG 是检索增强生成...", "RAG 的核心流程...", "另一种文档的内容"]],
    #     "metadatas": [[{"doc_id": "a"}, {"doc_id": "a"}, {"doc_id": "b"}]],
    # }
    distances = result["distances"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]

    for i in range(len(ids[0])):
        # distance 可能是 numpy.float32，转成原生 float 方便后续 JSON 序列化
        distance = float(distances[i])
        hits.append({
            "doc_id": metadatas[i].get("doc_id"),
            "chunk": documents[i],
            "distance": distance,
            "similarity": 1.0 - distance,
            "metadata": metadatas[i],
        })

    return hits

def delete_by_doc(doc_id: str) -> None:
    """删除某个 doc 的所有 chunk；不存在时静默通过（幂等）"""
    collection = _get_collection()
    collection.delete(where={"doc_id": doc_id})

def count() -> int:
    """返回 collection 内总条数，用于自检/监控"""
    return _get_collection().count()
