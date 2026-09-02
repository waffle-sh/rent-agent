"""전세 위험 판단 규칙. 외부 의존 없음 — LLM/HTTP 금지.

기준 출처 요약 (상세: docs/adr/0004-jeonse-risk-rules.md)
- 전세가율 90%: HUG 전세보증금반환보증 가입 요건 (2023.5.1~)
- 전세가율 70%: 업계 통용 안전권
- 소액임차인 표: 주택임대차보호법 시행령 제10조·제11조 (2023.2.21 개정)
- 주거비 30% 규칙: 월 주거비가 월소득의 30%를 넘으면 부담 과중으로 보는 통용 기준
"""

from rent_agent.domain.models import JeonseInput, Region, RiskAssessment, RiskLevel

# (보증금 상한, 최우선변제 한도) 단위: 만원
SMALL_TENANT_TABLE: dict[Region, tuple[int, int]] = {
    Region.SEOUL: (16500, 5500),
    Region.METRO_OVER: (14500, 4800),
    Region.METRO_CITY: (8500, 2800),
    Region.OTHER: (7500, 2500),
}

JEONSE_RATIO_SAFE = 70.0
JEONSE_RATIO_CAUTION = 80.0
JEONSE_RATIO_HUG_LIMIT = 90.0
BURDEN_CAUTION = 80.0
BURDEN_DANGER = 90.0
BURDEN_CRITICAL = 100.0
HOUSING_COST_INCOME_LIMIT = 30.0


def jeonse_ratio(deposit: int, market_price: int) -> float:
    return deposit / market_price * 100


def total_burden_ratio(
    deposit: int, senior_liens: int, senior_deposits: int, market_price: int
) -> float:
    return (deposit + senior_liens + senior_deposits) / market_price * 100


def expected_recovery(
    market_price: int,
    auction_ratio: float,
    senior_liens: int,
    senior_deposits: int,
    deposit: int,
    priority_amount: int = 0,
) -> tuple[int, int]:
    """경매 시 (회수 가능액, 부족액).

    배당 순서 가정: ① 소액임차인 최우선변제액(낙찰가의 1/2 한도) → ② 선순위 근저당·보증금
    → ③ 내 보증금 잔여분. 낙찰가는 보수적으로 절사한다.
    """
    proceeds = int(market_price * auction_ratio)
    priority = min(priority_amount, proceeds // 2)
    remainder = max(0, proceeds - priority - senior_liens - senior_deposits)
    recovery = min(deposit, priority + remainder)
    return recovery, deposit - recovery


def small_tenant_protection(region: Region, deposit: int) -> tuple[bool, int]:
    limit, priority = SMALL_TENANT_TABLE[region]
    if deposit > limit:
        return False, 0
    return True, min(priority, deposit)


def classify(jr: float, burden: float, shortfall: int) -> RiskLevel:
    if jr > JEONSE_RATIO_HUG_LIMIT or burden > BURDEN_CRITICAL:
        return RiskLevel.CRITICAL
    if jr > JEONSE_RATIO_CAUTION or burden > BURDEN_DANGER or shortfall > 0:
        return RiskLevel.DANGER
    if jr > JEONSE_RATIO_SAFE or burden > BURDEN_CAUTION:
        return RiskLevel.CAUTION
    return RiskLevel.SAFE


def assess(inp: JeonseInput) -> RiskAssessment:
    jr = jeonse_ratio(inp.deposit, inp.market_price)
    burden = total_burden_ratio(
        inp.deposit, inp.senior_liens, inp.senior_deposits, inp.market_price
    )
    protected, priority_amount = small_tenant_protection(inp.region, inp.deposit)
    recovery, shortfall = expected_recovery(
        inp.market_price,
        inp.auction_ratio,
        inp.senior_liens,
        inp.senior_deposits,
        inp.deposit,
        priority_amount=priority_amount,
    )
    required_loan = max(0, inp.deposit - inp.own_capital)
    monthly_interest = required_loan * inp.loan_rate / 100 / 12
    # 비교는 반올림 전 값으로, 표시는 소수 1자리로 (30.04% → 표시 30.0, 경고는 발생)
    ratio_raw = (
        monthly_interest / (inp.annual_income / 12) * 100 if inp.annual_income else None
    )
    ratio_to_income = round(ratio_raw, 1) if ratio_raw is not None else None
    level = classify(jr, burden, shortfall)

    reasons: list[str] = []
    reasons.append(
        f"전세가율 {jr:.1f}% (안전권 {JEONSE_RATIO_SAFE:.0f}% 이하, "
        f"HUG 보증 한도 {JEONSE_RATIO_HUG_LIMIT:.0f}%)"
    )
    if inp.senior_liens or inp.senior_deposits:
        reasons.append(
            f"선순위 근저당 {inp.senior_liens:,}만원·선순위 보증금 "
            f"{inp.senior_deposits:,}만원 포함 총 부담률 {burden:.1f}%"
        )
    if shortfall > 0:
        reasons.append(
            f"낙찰가율 {inp.auction_ratio:.0%} 가정 경매 시 회수 가능액 {recovery:,}만원, "
            f"보증금 대비 {shortfall:,}만원 부족"
        )
    else:
        reasons.append(f"낙찰가율 {inp.auction_ratio:.0%} 가정 경매 시에도 보증금 전액 회수 가능")
    if protected:
        reasons.append(
            f"소액임차인 최우선변제 대상: 최대 {priority_amount:,}만원을 선순위 채권보다 "
            "먼저 변제 (위 회수액 계산에 반영, 낙찰가의 1/2 한도)"
        )
    else:
        reasons.append("보증금이 소액임차인 기준을 초과하여 최우선변제 대상 아님")
    if required_loan:
        reasons.append(
            f"필요 대출 {required_loan:,}만원, 금리 {inp.loan_rate:g}% 기준 "
            f"월 이자 약 {monthly_interest:,.1f}만원"
        )
    if ratio_raw is not None and ratio_raw > HOUSING_COST_INCOME_LIMIT:
        reasons.append(
            f"월 이자가 월소득의 {ratio_to_income}%로 권고 상한 "
            f"{HOUSING_COST_INCOME_LIMIT:.0f}% 초과"
        )

    return RiskAssessment(
        jeonse_ratio=round(jr, 1),
        total_burden_ratio=round(burden, 1),
        expected_recovery=recovery,
        shortfall=shortfall,
        small_tenant_protected=protected,
        small_tenant_priority_amount=priority_amount,
        required_loan=required_loan,
        monthly_interest=round(monthly_interest, 1),
        interest_to_income_ratio=ratio_to_income,
        level=level,
        reasons=reasons,
    )
