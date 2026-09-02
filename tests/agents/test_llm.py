import os

import pytest

from rent_agent.agents.llm import configure_tracing, get_llm
from rent_agent.config import Settings


@pytest.fixture(autouse=True)
def _clean_langsmith_env():
    """configure_tracing()는 os.environ.setdefault()로 실제 프로세스 환경을 바꾼다
    (LangSmith 트레이서가 os.environ만 읽으므로 의도된 동작). monkeypatch는 이 직접
    쓰기를 추적하지 못해 테스트 종료 후에도 값이 남아 다른 테스트 파일(tests/test_config.py)을
    오염시킬 수 있으므로, 이 모듈의 각 테스트 뒤에 명시적으로 지운다."""
    yield
    for k in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        os.environ.pop(k, None)


def test_get_llm_uses_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    llm = get_llm(Settings(), temperature=0.3)
    assert llm.model_name == "gpt-test"
    assert llm.temperature == 0.3


def test_configure_tracing_exports_env_when_enabled(monkeypatch):
    for k in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        monkeypatch.delenv(k, raising=False)
    s = Settings(langsmith_tracing=True, langsmith_api_key="lsv2_test", langsmith_project="p")
    configure_tracing(s)
    import os

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2_test"
    assert os.environ["LANGSMITH_PROJECT"] == "p"


def test_configure_tracing_noop_when_disabled_or_no_key(monkeypatch):
    import os

    for k in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        monkeypatch.delenv(k, raising=False)
    configure_tracing(Settings(langsmith_tracing=True, langsmith_api_key=None))
    configure_tracing(Settings(langsmith_tracing=False, langsmith_api_key="lsv2_test"))
    assert "LANGSMITH_API_KEY" not in os.environ
