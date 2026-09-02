from datetime import date
from pathlib import Path

import httpx
import pytest

from rent_agent.tools.molit_rent import (
    HOUSING_SPECS,
    HousingType,
    MockMolitRentClient,
    MolitApiError,
    MolitRentClient,
    RentRecord,
    parse_rent_xml,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
ENDPOINTS = {
    HousingType.APARTMENT: "https://apis.data.go.kr/1613000/RTMSDataSvcAptRent",
    HousingType.MULTI_HOUSE: "https://apis.data.go.kr/1613000/RTMSDataSvcRHRent",
    HousingType.OFFICETEL: "https://apis.data.go.kr/1613000/RTMSDataSvcOffiRent",
}


def _xml(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parse_apartment_records_and_total():
    records, total = parse_rent_xml(_xml("rent_response.xml"), HousingType.APARTMENT)
    assert total == 1265
    assert len(records) == 4
    assert records[0] == RentRecord(
        housing_type=HousingType.APARTMENT,
        building_name="디에이치자이개포",
        sub_type="",
        dong="일원동",
        area_m2=76.46,
        floor=12,
        build_year=2021,
        deal_date=date(2026, 7, 24),
        deposit=100000,
        monthly_rent=200,
        contract_type="",
        renewal_right_used=False,
    )


def test_parse_handles_comma_deposit_and_renewal_flag():
    records, _ = parse_rent_xml(_xml("rent_response.xml"), HousingType.APARTMENT)
    renewed = records[2]
    assert renewed.deposit == 45000
    assert renewed.monthly_rent == 0
    assert renewed.is_jeonse is True
    assert renewed.renewal_right_used is True
    assert renewed.contract_type == "갱신"


def test_parse_multi_house_uses_mhouseNm_and_houseType():
    records, total = parse_rent_xml(_xml("rent_response_rh.xml"), HousingType.MULTI_HOUSE)
    assert total > 0 and len(records) == 4
    first = records[0]
    assert first.housing_type == HousingType.MULTI_HOUSE
    assert first.building_name == "개포비버리하임"
    assert first.sub_type == "다세대"
    assert first.deposit == 31300
    jeonse = [r for r in records if r.is_jeonse]
    assert sorted(r.deposit for r in jeonse) == [50000, 52500]


def test_parse_officetel_uses_offiNm():
    records, _ = parse_rent_xml(_xml("rent_response_offi.xml"), HousingType.OFFICETEL)
    assert len(records) == 4
    assert records[0].housing_type == HousingType.OFFICETEL
    assert records[0].building_name == "강남 지웰홈스"
    assert records[0].sub_type == ""
    assert not any(r.is_jeonse for r in records)


def test_parse_error_response_raises():
    with pytest.raises(MolitApiError, match="SERVICE_KEY_IS_NOT_REGISTERED_ERROR"):
        parse_rent_xml(_xml("rent_error.xml"), HousingType.APARTMENT)


def test_parse_empty_items():
    xml = (
        "<response><header><resultCode>000</resultCode><resultMsg>OK</resultMsg></header>"
        "<body><items/><numOfRows>10</numOfRows><pageNo>1</pageNo>"
        "<totalCount>0</totalCount></body></response>"
    )
    records, total = parse_rent_xml(xml, HousingType.APARTMENT)
    assert records == [] and total == 0


def test_client_sends_decoded_key_once_and_uses_operation_per_type():
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(str(request.url))
        body = "rent_response_rh.xml" if "RHRent" in str(request.url) else "rent_response.xml"
        return httpx.Response(200, text=_xml(body))

    client = MolitRentClient(
        ENDPOINTS,
        service_key="abc+def==",
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    apt = client.fetch(lawd_cd="11680", deal_ymd="202607")
    rh = client.fetch(lawd_cd="11680", deal_ymd="202607", housing_type=HousingType.MULTI_HOUSE)

    assert len(apt) == 4 and apt[0].housing_type == HousingType.APARTMENT
    assert len(rh) == 4 and rh[0].housing_type == HousingType.MULTI_HOUSE
    assert captured[0].startswith(
        "https://apis.data.go.kr/1613000/RTMSDataSvcAptRent/getRTMSDataSvcAptRent?"
    )
    assert captured[1].startswith(
        "https://apis.data.go.kr/1613000/RTMSDataSvcRHRent/getRTMSDataSvcRHRent?"
    )
    # httpx가 한 번만 인코딩: '+' → %2B, '=' → %3D
    assert "serviceKey=abc%2Bdef%3D%3D" in captured[0]
    assert "LAWD_CD=11680" in captured[0] and "DEAL_YMD=202607" in captured[0]


def test_client_http_error_wrapped():
    transport = httpx.MockTransport(lambda req: httpx.Response(500, text="boom"))
    client = MolitRentClient(ENDPOINTS, service_key="k", http=httpx.Client(transport=transport))
    with pytest.raises(MolitApiError):
        client.fetch(lawd_cd="11680", deal_ymd="202607")


def test_client_missing_endpoint_for_type():
    client = MolitRentClient({HousingType.APARTMENT: "https://x"}, service_key="k")
    with pytest.raises(MolitApiError, match="officetel"):
        client.fetch("11680", "202607", housing_type=HousingType.OFFICETEL)


@pytest.mark.parametrize("housing_type", list(HousingType))
def test_mock_client_returns_fixture_per_type(housing_type):
    records = MockMolitRentClient().fetch(
        lawd_cd="11680", deal_ymd="202607", housing_type=housing_type
    )
    assert len(records) == 4
    assert all(r.housing_type == housing_type for r in records)


def test_housing_specs_cover_all_types():
    assert set(HOUSING_SPECS) == set(HousingType)
