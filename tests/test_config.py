from rent_agent.config import Settings, get_settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abc")
    monkeypatch.setenv("APARTMENT_OPENAPI_KEY", "abc%2Bdef%3D%3D")
    s = Settings(_env_file=None)
    assert s.openai_api_key == "sk-abc"
    assert s.openai_model == "gpt-4.1-mini"
    assert s.apartment_openapi_endpoint.endswith("/RTMSDataSvcAptRent")
    assert s.multi_house_openapi_endpoint.endswith("/RTMSDataSvcRHRent")
    assert s.office_openapi_endpoint.endswith("/RTMSDataSvcOffiRent")


def test_apartment_key_is_url_decoded(monkeypatch):
    """data.go.kr 'Encoding 키'를 그대로 넣어도 httpx가 재인코딩하지 않도록 디코딩된 값을 제공."""
    monkeypatch.setenv("APARTMENT_OPENAPI_KEY", "abc%2Bdef%3D%3D")
    s = Settings(_env_file=None)
    assert s.apartment_openapi_key_decoded == "abc+def=="


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_real_env_file_is_not_read_in_tests():
    """conftest가 RENT_AGENT_ENV_FILE을 없는 경로로 고정하므로,
    .env에만 있는 값은 비어 있어야 한다."""
    s = Settings()
    assert s.langsmith_api_key is None
    assert s.apartment_openapi_key == "test%2Bkey%3D%3D"  # conftest 더미
