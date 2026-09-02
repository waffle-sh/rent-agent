import os
from pathlib import Path

import pytest

# rent_agent.config가 import될 때 실제 .env 대신 존재하지 않는 파일을
# 보게 한다 (fixture보다 먼저 실행되어야 함).
os.environ["RENT_AGENT_ENV_FILE"] = str(Path(__file__).parent / "does-not-exist.env")


@pytest.fixture(autouse=True)
def _dummy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """유닛 테스트는 실제 키 없이 돌아야 한다.
    - 필수 키는 더미로 채운다.
    - .env 파일은 위 RENT_AGENT_ENV_FILE 덕분에 읽히지 않는다.
    - get_settings()의 lru_cache를 매 테스트 전에 비워 테스트 간 오염을 막는다."""
    from rent_agent.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("APARTMENT_OPENAPI_KEY", "test%2Bkey%3D%3D")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("MOLIT_USE_MOCK", "true")
