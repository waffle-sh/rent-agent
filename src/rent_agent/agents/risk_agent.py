"""위험 판단 에이전트: 계산은 domain.risk(결정적), LLM은 입력 추출과 설명만."""

from typing import Literal

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import ValidationError

from rent_agent.agents.llm import get_llm
from rent_agent.agents.prompts import RISK_PROMPT
from rent_agent.config import Settings
from rent_agent.domain.models import JeonseInput, Region
from rent_agent.domain.risk import assess

# 100억 만원 = 1조원. 이보다 크면 LLM이 원 단위를 넣은 것 (가장 흔한 추출 실수)
MAX_REASONABLE_MANWON = 1_000_000

_ERROR_KO = {
    "greater_than": "0보다 커야 합니다",
    "greater_than_equal": "0 이상이어야 합니다",
    "less_than_equal": "상한을 넘었습니다",
    "int_parsing": "정수여야 합니다",
    "float_parsing": "숫자여야 합니다",
}


def _format_validation_error(e: ValidationError) -> str:
    """pydantic 오류를 LLM이 바로 고칠 수 있는 짧은 한국어로.

    예: 'market_price: 0보다 커야 합니다'
    """
    parts = []
    for err in e.errors():
        field = ".".join(str(x) for x in err["loc"])
        parts.append(f"{field}: {_ERROR_KO.get(err['type'], err['msg'])}")
    return "; ".join(parts)


@tool
def assess_jeonse_risk(
    deposit: int,
    market_price: int,
    senior_liens: int = 0,
    senior_deposits: int = 0,
    region: Literal["seoul", "metro_over", "metro_city", "other"] = "seoul",
    own_capital: int = 0,
    annual_income: int | None = None,
    loan_rate: float = 3.5,
) -> str:
    """전세 계약의 위험도를 규칙 기반으로 계산한다. 모든 금액은 '만원' 단위 (3억 → 30000).

    deposit: 전세 보증금. market_price: 해당 주택 매매 시세.
    senior_liens: 등기부 을구 선순위 근저당 채권최고액 합계.
    senior_deposits: 선순위 임차보증금 합계.
    region: 소액임차인 기준 지역 (find_region_code의 small_tenant_region 값을 그대로 사용).
    own_capital: 자기자금. annual_income: 연소득(선택). loan_rate: 전세대출 예상 금리(%).
    반환: 전세가율, 총 부담률, 경매 시 회수액/부족액, 소액임차인 여부, 필요 대출·월 이자,
    판정(안전/주의/위험/매우 위험), 근거 문장 목록 (JSON).
    """
    if max(deposit, market_price) > MAX_REASONABLE_MANWON:
        return (
            f"입력 오류: 금액이 비정상적으로 큽니다 (deposit={deposit:,}, "
            f"market_price={market_price:,}). "
            "원이 아닌 만원 단위로 변환해 다시 호출하세요 (예: 3억 5천만원 → 35000)."
        )
    try:
        inp = JeonseInput(
            deposit=deposit,
            market_price=market_price,
            senior_liens=senior_liens,
            senior_deposits=senior_deposits,
            region=Region(region),
            own_capital=own_capital,
            annual_income=annual_income,
            loan_rate=loan_rate,
        )
    except ValidationError as e:
        return (
            f"입력 오류: {_format_validation_error(e)}. "
            "값을 확인해 다시 호출하세요 (금액은 만원 단위)."
        )
    return assess(inp).model_dump_json()


def build_risk_agent(settings: Settings):
    return create_agent(
        model=get_llm(settings),
        tools=[assess_jeonse_risk],
        system_prompt=RISK_PROMPT,
        name="risk_agent",
    )
