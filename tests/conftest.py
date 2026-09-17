import pytest
#名字必须为conftest.py，否则不会被pytest识别
@pytest.fixture(scope="session")
def base_url():
    """
    pytest 会话级 fixture，提供被测服务的基础 URL。
    整个测试会话内只执行一次，所有测试用例共享同一个返回值。
    支持通过环境变量 TEST_BASE_URL 覆盖默认地址，方便在不同环境（本地/CI/远程）之间切换。
    """
    import os
    # 优先读取环境变量 TEST_BASE_URL，未设置时回退到默认本地地址
    return os.getenv("TEST_BASE_URL", "http://localhost:8000")