import pytest


@pytest.fixture(autouse=True)
def _dummy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """유닛 테스트는 실제 키 없이 돌아야 한다. .env 파일도 읽지 않게 한다."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("APARTMENT_OPENAPI_KEY", "test%2Bkey%3D%3D")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("MOLIT_USE_MOCK", "true")
