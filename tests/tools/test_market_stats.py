from datetime import date

from rent_agent.tools.market_stats import JeonseMarketSummary, summarize_jeonse
from rent_agent.tools.molit_rent import HousingType, RentRecord


def rec(apt="까치마을", area=39.6, deposit=40000, rent=0, day=1) -> RentRecord:
    return RentRecord(
        housing_type=HousingType.APARTMENT,
        building_name=apt,
        sub_type="",
        dong="수서동",
        area_m2=area,
        floor=5,
        build_year=1993,
        deal_date=date(2026, 7, day),
        deposit=deposit,
        monthly_rent=rent,
        contract_type="신규",
        renewal_right_used=False,
    )


def test_filters_pure_jeonse_and_apt_and_area():
    records = [
        rec(deposit=40000, day=1),
        rec(deposit=45000, day=2),
        rec(deposit=50000, day=3),
        rec(deposit=20000, rent=90),  # 월세 → 제외
        rec(apt="다른단지", deposit=99999),  # 단지 다름 → 제외
        rec(area=59.9, deposit=70000),  # 면적 차이 > 허용치 → 제외
    ]
    s = summarize_jeonse(records, building_name="까치마을", area_m2=39.6, area_tolerance=5.0)
    assert s.count == 3
    assert s.median_deposit == 45000
    assert s.min_deposit == 40000 and s.max_deposit == 50000
    assert [r.deposit for r in s.recent] == [50000, 45000, 40000]  # 최신순


def test_building_name_partial_match_and_no_area_filter():
    records = [rec(apt="까치마을1단지"), rec(apt="까치마을2단지", area=59.9)]
    s = summarize_jeonse(records, building_name="까치마을")
    assert s.count == 2


def test_empty_summary():
    s = summarize_jeonse([], building_name="없는단지")
    assert s == JeonseMarketSummary(
        count=0, median_deposit=None, min_deposit=None, max_deposit=None, recent=[]
    )


def test_compare_ratio():
    s = summarize_jeonse([rec(deposit=40000), rec(deposit=50000)])
    assert s.median_deposit == 45000
    assert s.ratio_to_median(54000) == 120.0
