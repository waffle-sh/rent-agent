import pytest

from rent_agent.domain.models import JeonseInput, Region, RiskLevel
from rent_agent.domain.risk import (
    assess,
    classify,
    expected_recovery,
    jeonse_ratio,
    small_tenant_protection,
    total_burden_ratio,
)


def test_jeonse_ratio_percent():
    assert jeonse_ratio(deposit=35000, market_price=50000) == pytest.approx(70.0)


def test_total_burden_includes_senior_claims():
    # 근저당 1억 + 선순위 보증금 5천 + 내 보증금 3억 / 매매가 5억 = 90%
    assert total_burden_ratio(
        deposit=30000, senior_liens=10000, senior_deposits=5000, market_price=50000
    ) == pytest.approx(90.0)


def test_expected_recovery_and_shortfall():
    # 5억 * 0.8 = 4억 낙찰, 선순위 1억 → 회수 가능 3억, 보증금 3.5억 → 부족 5천
    recovery, shortfall = expected_recovery(
        market_price=50000, auction_ratio=0.8, senior_liens=10000, senior_deposits=0, deposit=35000
    )
    assert recovery == 30000
    assert shortfall == 5000


def test_expected_recovery_no_shortfall_is_zero():
    _, shortfall = expected_recovery(
        market_price=50000, auction_ratio=0.8, senior_liens=0, senior_deposits=0, deposit=30000
    )
    assert shortfall == 0


@pytest.mark.parametrize(
    "region,deposit,eligible,amount",
    [
        (Region.SEOUL, 16500, True, 5500),
        (Region.SEOUL, 16501, False, 0),
        (Region.METRO_OVER, 14500, True, 4800),
        (Region.METRO_CITY, 8500, True, 2800),
        (Region.OTHER, 7500, True, 2500),
        (Region.OTHER, 9000, False, 0),
    ],
)
def test_small_tenant_protection(region, deposit, eligible, amount):
    assert small_tenant_protection(region, deposit) == (eligible, amount)


def test_small_tenant_priority_capped_by_deposit():
    # 보증금 3천만이면 우선변제액도 3천만 (5,500만 아님)
    assert small_tenant_protection(Region.SEOUL, 3000) == (True, 3000)


@pytest.mark.parametrize(
    "jr,tb,shortfall,expected",
    [
        (60.0, 60.0, 0, RiskLevel.SAFE),
        (75.0, 75.0, 0, RiskLevel.CAUTION),
        (65.0, 85.0, 0, RiskLevel.CAUTION),
        (85.0, 85.0, 0, RiskLevel.DANGER),
        (65.0, 65.0, 1000, RiskLevel.DANGER),
        (95.0, 95.0, 0, RiskLevel.CRITICAL),
        (60.0, 105.0, 0, RiskLevel.CRITICAL),
    ],
)
def test_classify(jr, tb, shortfall, expected):
    assert classify(jr, tb, shortfall) == expected


def test_assess_full_case_danger():
    inp = JeonseInput(
        deposit=35000,
        market_price=50000,
        senior_liens=10000,
        region=Region.SEOUL,
        own_capital=15000,
        annual_income=4800,
        loan_rate=4.0,
    )
    result = assess(inp)
    assert result.jeonse_ratio == pytest.approx(70.0)
    assert result.total_burden_ratio == pytest.approx(90.0)
    assert result.shortfall == 5000
    assert result.level == RiskLevel.DANGER
    assert result.small_tenant_protected is False
    assert result.required_loan == 20000
    assert result.monthly_interest == pytest.approx(20000 * 0.04 / 12, abs=1)
    # 월소득 400만, 월이자 약 66.7만 → 약 16.7%
    assert result.interest_to_income_ratio == pytest.approx(16.7, abs=0.1)
    assert any("근저당" in r or "선순위" in r for r in result.reasons)


def test_assess_without_income_has_none_ratio():
    inp = JeonseInput(deposit=20000, market_price=50000)
    result = assess(inp)
    assert result.interest_to_income_ratio is None
    assert result.level == RiskLevel.SAFE


def test_assess_interest_burden_adds_reason():
    # 보증금 3억 전부 대출, 금리 5%, 연소득 3천 → 월이자 125만 / 월소득 250만 = 50%
    inp = JeonseInput(
        deposit=30000, market_price=60000, own_capital=0, annual_income=3000, loan_rate=5.0
    )
    result = assess(inp)
    assert result.interest_to_income_ratio == pytest.approx(50.0)
    assert any("30%" in r for r in result.reasons)


def test_input_validation_rejects_zero_price():
    with pytest.raises(ValueError):
        JeonseInput(deposit=1000, market_price=0)
