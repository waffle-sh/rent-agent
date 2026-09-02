import json
from datetime import date

from rent_agent.agents.market_agent import (
    make_market_tools,
    recent_deal_months,
    small_tenant_region,
)
from rent_agent.tools.molit_rent import HousingType, MockMolitRentClient, MolitApiError


def test_recent_deal_months():
    assert recent_deal_months(today=date(2026, 9, 2), months=3) == ["202609", "202608", "202607"]
    assert recent_deal_months(today=date(2026, 1, 15), months=2) == ["202601", "202512"]


def test_find_region_code_tool_includes_small_tenant_region():
    find_region_code, _ = make_market_tools(MockMolitRentClient())
    out = json.loads(find_region_code.invoke({"query": "강남구"}))
    assert out == [{"name": "서울특별시 강남구", "code": "11680", "small_tenant_region": "seoul"}]
    out = json.loads(find_region_code.invoke({"query": "분당"}))
    assert out[0]["small_tenant_region"] == "metro_over"


def test_get_recent_jeonse_deals_apartment_default():
    _, get_recent_jeonse_deals = make_market_tools(MockMolitRentClient())
    out = json.loads(
        get_recent_jeonse_deals.invoke(
            {"lawd_cd": "11680", "building_name": "까치마을", "area_m2": 39.6, "months": 1}
        )
    )
    assert out["housing_type"] == "apartment"
    assert out["count"] == 1  # 픽스처: 까치마을 39.6㎡ 순수 전세 1건 (45,000) — 갱신 계약
    assert out["median_deposit"] == 45000
    assert out["new_contract_count"] == 0 and out["new_contract_median"] is None
    assert out["recent"][0]["building_name"] == "까치마을"
    assert out["recent"][0]["deal_date"] == "2026-07-10"


def test_get_recent_jeonse_deals_multi_house():
    _, get_recent_jeonse_deals = make_market_tools(MockMolitRentClient())
    out = json.loads(
        get_recent_jeonse_deals.invoke(
            {"lawd_cd": "11680", "housing_type": "multi_house", "months": 1}
        )
    )
    assert out["housing_type"] == "multi_house"
    assert out["count"] == 2  # RH 픽스처 순수 전세 52,500 / 50,000 (계약구분 미기재 → 신규 간주)
    assert out["median_deposit"] == 51250
    assert out["new_contract_count"] == 2 and out["new_contract_median"] == 51250
    assert out["recent"][0]["sub_type"] in ("연립", "다세대")


def test_reference_and_ratio_computed_by_tool():
    _, get_recent_jeonse_deals = make_market_tools(MockMolitRentClient())
    # 아파트 까치마을 39.6: 신규 0건 → 기준은 전체 중위값 45,000 (갱신 포함). 보증금 54,000 → 120.0%
    out = json.loads(
        get_recent_jeonse_deals.invoke(
            {
                "lawd_cd": "11680",
                "building_name": "까치마을",
                "area_m2": 39.6,
                "months": 1,
                "deposit": 54000,
            }
        )
    )
    assert out["reference_median"] == 45000
    assert out["reference_basis"] == "전체(갱신 포함, 신규 3건 미만)"
    assert out["ratio_to_reference"] == 120.0
    # 보증금 미제공 → 비율 없음
    out2 = json.loads(
        get_recent_jeonse_deals.invoke(
            {"lawd_cd": "11680", "building_name": "까치마을", "months": 1}
        )
    )
    assert out2["ratio_to_reference"] is None and out2["reference_median"] is not None


def test_get_recent_jeonse_deals_officetel_no_jeonse():
    _, get_recent_jeonse_deals = make_market_tools(MockMolitRentClient())
    out = json.loads(
        get_recent_jeonse_deals.invoke(
            {"lawd_cd": "11680", "housing_type": "officetel", "months": 1}
        )
    )
    assert out["count"] == 0 and "message" in out


def test_housing_type_is_an_enum_in_tool_schema():
    _, get_recent_jeonse_deals = make_market_tools(MockMolitRentClient())
    props = get_recent_jeonse_deals.args_schema.model_json_schema()["properties"]
    assert props["housing_type"]["enum"] == ["apartment", "multi_house", "officetel"]


def test_months_is_clamped_and_errors_are_reported():
    class FlakyClient:
        calls: list[str] = []

        def fetch(self, lawd_cd, deal_ymd, housing_type=HousingType.APARTMENT, num_of_rows=1000):
            self.calls.append(deal_ymd)
            if deal_ymd.endswith("08"):
                raise MolitApiError("LIMITED NUMBER OF SERVICE REQUESTS EXCEEDS")
            return MockMolitRentClient().fetch(lawd_cd, deal_ymd, housing_type)

    client = FlakyClient()
    _, get_recent_jeonse_deals = make_market_tools(client)
    out = json.loads(get_recent_jeonse_deals.invoke({"lawd_cd": "11680", "months": 24}))
    assert len(out["months_queried"]) == 12  # 24 → 12로 클램프
    assert len(client.calls) == 12
    assert any(
        e.endswith("08: LIMITED NUMBER OF SERVICE REQUESTS EXCEEDS") for e in out["errors"]
    )
    assert out["count"] > 0  # 실패한 달을 제외한 나머지는 요약됨

    out0 = json.loads(get_recent_jeonse_deals.invoke({"lawd_cd": "11680", "months": 0}))
    assert len(out0["months_queried"]) == 1  # 0 → 1


def test_reference_uses_new_contract_median_at_exactly_three():
    from datetime import date as _date

    from rent_agent.tools.molit_rent import RentRecord

    def rec(deposit, contract):
        return RentRecord(
            housing_type=HousingType.APARTMENT,
            building_name="X",
            sub_type="",
            dong="d",
            area_m2=59.9,
            floor=1,
            build_year=2000,
            deal_date=_date(2026, 7, 1),
            deposit=deposit,
            monthly_rent=0,
            contract_type=contract,
            renewal_right_used=contract == "갱신",
        )

    class StaticClient:
        def fetch(self, lawd_cd, deal_ymd, housing_type=HousingType.APARTMENT, num_of_rows=1000):
            return [rec(50000, "신규"), rec(52000, "신규"), rec(54000, ""), rec(40000, "갱신")]

    _, get_recent_jeonse_deals = make_market_tools(StaticClient())
    out = json.loads(
        get_recent_jeonse_deals.invoke({"lawd_cd": "11680", "months": 1, "deposit": 52000})
    )
    assert out["new_contract_count"] == 3
    assert out["reference_basis"] == "신규 계약 중위값" and out["reference_median"] == 52000
    assert out["ratio_to_reference"] == 100.0


def test_small_tenant_region_allowlist_covers_every_code():
    # 정적 함수(서울→seoul, 그 외→metro_over)가 유효한 범위를 데이터로 고정.
    # 코드표에 파주·인천 등을 추가하면 이 테스트가 먼저 깨진다.
    from rent_agent.tools.lawd_code import LAWD_CODES

    metro_over_cities = (
        "수원시",
        "성남시",
        "고양시",
        "용인시",
        "부천시",
        "안양시",
        "화성시",
        "하남시",
        "광명시",
        "과천시",
    )
    for name in LAWD_CODES:
        if name.startswith("서울"):
            assert small_tenant_region(name) == "seoul"
        else:
            assert any(city in name for city in metro_over_cities), name
            assert small_tenant_region(name) == "metro_over"
