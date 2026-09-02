from datetime import date

from rent_agent.tools.market_stats import JeonseMarketSummary, summarize_jeonse
from rent_agent.tools.molit_rent import HousingType, RentRecord


def rec(apt="까치마을", area=39.6, deposit=40000, rent=0, day=1, contract="신규") -> RentRecord:
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
        contract_type=contract,
        renewal_right_used=contract == "갱신",
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


def test_building_name_ignores_whitespace_and_case():
    # 실제 API 표기 "강남 지웰홈스"(공백 포함)를 "강남지웰홈스"로 찾을 수 있어야 한다
    records = [rec(apt="강남 지웰홈스"), rec(apt="Raemian Blesstige")]
    assert summarize_jeonse(records, building_name="강남지웰홈스").count == 1
    assert summarize_jeonse(records, building_name=" 강남  지웰홈스 ").count == 1
    assert summarize_jeonse(records, building_name="raemian").count == 1


def test_empty_building_name_means_no_filter():
    records = [rec(apt="A"), rec(apt="B")]
    assert summarize_jeonse(records, building_name="").count == 2
    assert summarize_jeonse(records, building_name="   ").count == 2


def test_new_contract_median_excludes_renewals():
    # 갱신 계약은 5% 상한 때문에 2년 전 가격 → 시세 신호가 아님.
    # 전체 중위값과 별도로 신규 중위값 제공
    records = [
        rec(deposit=40000, contract="갱신"),
        rec(deposit=41000, contract="갱신"),
        rec(deposit=50000, contract="신규"),
        rec(deposit=52000, contract=""),  # 계약구분 미기재(2021년 이전 등)는 신규로 간주
    ]
    s = summarize_jeonse(records)
    assert s.count == 4 and s.median_deposit == 45500
    assert s.new_contract_count == 2
    assert s.new_contract_median == 51000


def test_new_contract_median_none_when_all_renewals():
    s = summarize_jeonse([rec(deposit=40000, contract="갱신")])
    assert s.count == 1 and s.new_contract_count == 0 and s.new_contract_median is None


def test_empty_summary():
    s = summarize_jeonse([], building_name="없는단지")
    assert s == JeonseMarketSummary(
        count=0, median_deposit=None, min_deposit=None, max_deposit=None, recent=[]
    )
    assert s.ratio_to_median(30000) is None


def test_recent_ties_broken_by_deposit_desc():
    records = [rec(deposit=40000, day=3), rec(deposit=48000, day=3), rec(deposit=44000, day=3)]
    assert [r.deposit for r in summarize_jeonse(records).recent] == [48000, 44000, 40000]


def test_compare_ratio():
    s = summarize_jeonse([rec(deposit=40000), rec(deposit=50000)])
    assert s.median_deposit == 45000
    assert s.ratio_to_median(54000) == 120.0
