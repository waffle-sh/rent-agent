import json
from datetime import date

from rent_agent.agents.market_agent import make_market_tools, recent_deal_months
from rent_agent.tools.molit_rent import MockMolitRentClient


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


def test_get_recent_jeonse_deals_invalid_housing_type():
    _, get_recent_jeonse_deals = make_market_tools(MockMolitRentClient())
    out = json.loads(get_recent_jeonse_deals.invoke({"lawd_cd": "11680", "housing_type": "villa"}))
    assert "error" in out and "multi_house" in out["error"]
