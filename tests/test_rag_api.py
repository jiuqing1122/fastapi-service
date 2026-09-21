"""
RAG 上传与文档管理接口的集成测试。
FastAPI 服务已在 base_url（默认 http://localhost:8000）运行。
"""
import io
import uuid
import requests

def _txt_file(content: str = None, filename: str = None) -> dict:
    """构造一个 multipart 上传的 files 参数"""
    content = content or ("星海智能科技知识库测试内容" * 62)  # 13*62=806 > chunk_size(800)
    filename = filename or f"test_{uuid.uuid4().hex[:8]}.txt"
    return {"file": (filename, io.BytesIO(content.encode("utf-8")), "text/plain")}

#======上传：正常路径======
def test_upload_txt_success(base_url):
    resp = requests.post(f"{base_url}/ai/rag/upload", files=_txt_file())
    body = resp.json()
    assert resp.status_code == 200
    assert body["code"] == 200
    assert body["data"]["chunk_count"] > 1
    assert body["data"]["status"] == "ready"

    #清理
    doc_id = body["data"]["doc_id"]
    requests.delete(f"{base_url}/ai/rag/documents/{doc_id}")

#======上传：校验失败路径======
def test_upload_empty_file(base_url):
    """上传空文件"""
    resp = requests.post(
        f"{base_url}/ai/rag/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    )
    body = resp.json()
    assert body["code"] == 40002

def test_upload_bad_extension(base_url):
    """上传扩展名错误的文件"""
    resp = requests.post(
        f"{base_url}/ai/rag/upload",
        files={"file": ("evil.exe", io.BytesIO(b"MZ\x90\x00" + b"x" * 100), "application/octet-stream")},
    )
    body = resp.json()
    assert body["code"] == 40001

def test_upload_fake_pdf(base_url):
    """扩展名是 pdf，但内容是纯文本 → magic bytes 拒绝"""
    resp = requests.post(
        f"{base_url}/ai/rag/upload",
        files={"file": ("fake.pdf", io.BytesIO(b"this is not a pdf at all"), "application/pdf")},
    )
    body = resp.json()
    assert body["code"] == 40004

def test_upload_oversize(base_url):
    """构造 > 20MB 数据 → 40003"""
    big = b"a" * (21 *1024 * 1024)
    resp = requests.post(
        f"{base_url}/ai/rag/upload",
        files={"file": ("big.txt", io.BytesIO(big), "text/plain")},
    )
    body = resp.json()
    assert body["code"] == 40003

def test_upload_path_traversal(base_url):
    """路径穿越文件名 → 落盘名被 sanitize，不含 / 和 .."""
    resp = requests.post(
        f"{base_url}/ai/rag/upload",
        files=_txt_file(filename="../../etc/passwd.txt"),
    )
    body = resp.json()
    if body["code"] == 200:
        safe = body["data"]["filename"]
        assert "/" not in safe
        assert "\\" not in safe
        assert ".." not in safe
        requests.delete(f"{base_url}/ai/rag/documents/{body['data']['doc_id']}")

#======列表 / 详情 / 删除 ======
def test_list_get_delete_flow(base_url):
    # 1. 上传
    resp = requests.post(f"{base_url}/ai/rag/upload", files=_txt_file())
    doc_id = resp.json()["data"]["doc_id"]

    # 2. 列表里能查到
    resp = requests.get(f"{base_url}/ai/rag/documents?page=1&page_size=50")
    body = resp.json()
    assert body["code"] == 200
    #验证上传接口和列表接口的数据一致性
    assert any(m["doc_id"] == doc_id for m in body["data"]["items"])

    # 3. 按 keyword 过滤
    resp = requests.get(f"{base_url}/ai/rag/documents?keyword=test_")
    body = resp.json()
    assert body["code"] == 200
    assert all("test_" in m["filename"] for m in body["data"]["items"])

    # 4. 详情
    resp = requests.get(f"{base_url}/ai/rag/documents/{doc_id}")
    assert resp.json()["data"]["doc_id"] == doc_id

    # 5. 删除
    resp = requests.delete(f"{base_url}/ai/rag/documents/{doc_id}")
    assert resp.json()["code"] == 200
    assert resp.json()["data"]["existed"] is True

    # 6. 删除后查详情 → 40401
    resp = requests.get(f"{base_url}/ai/rag/documents/{doc_id}")
    assert resp.json()["code"] == 40401

    # 7. 重复删除 → 幂等，仍返回 200
    resp = requests.delete(f"{base_url}/ai/rag/documents/{doc_id}")
    assert resp.json()["code"] == 200
    assert resp.json()["data"]["existed"] is False
