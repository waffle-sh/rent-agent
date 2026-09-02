"""실행: uv run pytest -m integration tests/tools/test_lawd_code_live.py"""

import pytest

from rent_agent.config import PROJECT_ROOT, Settings
from rent_agent.tools.lawd_code import LAWD_CODES
from rent_agent.tools.molit_rent import HousingType, MolitRentClient

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("name,code", list(LAWD_CODES.items()))
def test_every_code_returns_apartment_rent_data(name, code):
    s = Settings(_env_file=PROJECT_ROOT / ".env")
    client = MolitRentClient(
        {HousingType.APARTMENT: s.apartment_openapi_endpoint}, s.apartment_openapi_key_decoded
    )
    # num_of_rows 기본값(1,000) 사용: 1로 두면 페이지네이션이 MAX_PAGES까지 돌아 20배 느려진다
    records = client.fetch(code, "202607")
    assert records, f"{name}({code}) 실거래 0건 — 행정구역 개편으로 코드가 바뀌었을 수 있음"
