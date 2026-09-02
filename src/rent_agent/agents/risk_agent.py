"""위험 판단 에이전트: 계산은 domain.risk(결정적), LLM은 입력 추출과 설명만."""

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import ValidationError

from rent_agent.agents.llm import get_llm
from rent_agent.agents.prompts import RISK_PROMPT
from rent_agent.config import Settings
from rent_agent.domain.models import JeonseInput, Region
from rent_agent.domain.risk import assess


@tool
def assess_jeonse_risk(
    deposit: int,
    market_price: int,
    senior_liens: int = 0,
    senior_deposits: int = 0,
    region: str = "seoul",
    own_capital: int = 0,
    annual_income: int | None = None,
    loan_rate: float = 3.5,
) -> str:
    """전세 계약의 위험도를 규칙 기반으로 계산한다. 모든 금액은 '만원' 단위.

    deposit: 전세 보증금. market_price: 해당 주택 매매 시세.
    senior_liens: 등기부 을구 선순위 근저당 채권최고액 합계.
    senior_deposits: 선순위 임차보증금 합계.
    region: seoul | metro_over | metro_city | other (소액임차인 기준 지역).
    own_capital: 자기자금. annual_income: 연소득(선택). loan_rate: 전세대출 예상 금리(%).
    반환: 전세가율, 총 부담률, 경매 시 회수액/부족액, 소액임차인 여부, 필요 대출·월 이자,
    판정(안전/주의/위험/매우 위험), 근거 문장 목록 (JSON).
    """
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
    except ValueError as e:  # ValidationError도 ValueError 하위
        valid = ", ".join(r.value for r in Region)
        detail = e.errors() if isinstance(e, ValidationError) else str(e)
        return f"입력 오류: {detail}. region은 다음 중 하나여야 합니다: {valid}"
    return assess(inp).model_dump_json()


def build_risk_agent(settings: Settings):
    return create_agent(
        model=get_llm(settings),
        tools=[assess_jeonse_risk],
        system_prompt=RISK_PROMPT,
        name="risk_agent",
    )
