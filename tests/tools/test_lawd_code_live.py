"""실행: uv run pytest -m integration tests/tools/test_lawd_code_live.py"""

import pytest

from rent_agent.config import PROJECT_ROOT, Settings
from rent_agent.tools.lawd_code import LAWD_CODES
from rent_agent.tools.molit_rent import HousingType, MolitRentClient

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _use_real_service_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """conftest의 autouse 픽스처가 넣어 둔 더미 키를 지운다.
    pydantic-settings는 환경변수를 .env보다 우선하므로, 지우지 않으면 실제 키가 무시된다."""
    monkeypatch.delenv("APARTMENT_OPENAPI_KEY", raising=False)


@pytest.mark.parametrize("name,code", list(LAWD_CODES.items()))
def test_every_code_returns_apartment_rent_data(name, code):
    s = Settings(_env_file=PROJECT_ROOT / ".env")
    client = MolitRentClient(
        {HousingType.APARTMENT: s.apartment_openapi_endpoint}, s.apartment_openapi_key_decoded
    )
    records = client.fetch(code, "202607", num_of_rows=1)
    assert records, f"{name}({code}) 실거래 0건 — 행정구역 개편으로 코드가 바뀌었을 수 있음"
