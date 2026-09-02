"""시세 조회 에이전트: 법정동코드 조회 + 실거래가 API(아파트/연립다세대/오피스텔) + 통계 요약."""

import json
from dataclasses import asdict
from datetime import date

from langchain.agents import create_agent
from langchain_core.tools import BaseTool, tool

from rent_agent.agents.llm import get_llm
from rent_agent.agents.prompts import MARKET_PROMPT
from rent_agent.config import Settings
from rent_agent.tools.lawd_code import find_lawd_codes
from rent_agent.tools.market_stats import summarize_jeonse
from rent_agent.tools.molit_rent import (
    HousingType,
    MockMolitRentClient,
    MolitApiError,
    MolitRentClient,
    RentClient,
)

MIN_NEW_CONTRACTS = 3  # 신규 계약이 이 건수 이상이면 신규 중위값을 시세 기준으로 쓴다


def small_tenant_region(region_name: str) -> str:
    """법정동 정식 명칭 → 소액임차인 최우선변제 지역 구분.

    현재 코드표(서울 25구 + 경기 25개 시·구)에만 유효: 경기 항목은 모두
    과밀억제권역(수원·성남·고양·부천·안양·하남·광명·과천) 또는 시행령이 직접
    명시한 도시(용인·화성)다."""
    return "seoul" if region_name.startswith("서울") else "metro_over"


def recent_deal_months(today: date | None = None, months: int = 3) -> list[str]:
    """오늘부터 과거 months개월의 YYYYMM 목록 (최신 먼저)."""
    today = today or date.today()
    out: list[str] = []
    y, m = today.year, today.month
    for _ in range(months):
        out.append(f"{y}{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return out


def make_market_tools(client: RentClient) -> tuple[BaseTool, BaseTool]:
    @tool
    def find_region_code(query: str) -> str:
        """지역명(예: '강남구', '분당구', '서울 강남구')으로 실거래가 조회에 필요한 시군구
        법정동코드(5자리)를 찾는다.
        부분 일치 목록을 JSON [{name, code, small_tenant_region}]로 반환. small_tenant_region은
        소액임차인 최우선변제 지역 구분(seoul | metro_over)으로, risk 판단 시 region 인자로
        그대로 쓴다. 동(洞) 이름은 지원하지 않는다."""
        return json.dumps(
            [
                {"name": n, "code": c, "small_tenant_region": small_tenant_region(n)}
                for n, c in find_lawd_codes(query)
            ],
            ensure_ascii=False,
        )

    @tool
    def get_recent_jeonse_deals(
        lawd_cd: str,
        housing_type: str = "apartment",
        building_name: str | None = None,
        area_m2: float | None = None,
        months: int = 3,
        deposit: int | None = None,
    ) -> str:
        """국토부 전월세 실거래가에서 최근 N개월 순수 전세(월세 0) 거래를 조회해 요약한다.
        lawd_cd: find_region_code로 얻은 5자리 코드.
        housing_type: apartment(아파트) | multi_house(연립·다세대·빌라) | officetel(오피스텔).
        building_name: 건물/단지명 일부(선택). area_m2: 전용면적 ㎡(선택, ±5㎡).
        deposit: 사용자의 보증금(만원, 선택).
        반환 JSON: housing_type, count, median_deposit(전체), new_contract_count,
        new_contract_median(갱신 제외), reference_median·reference_basis(도구가 정한 시세 기준값:
        신규 3건 이상이면 신규 중위값, 아니면 전체), ratio_to_reference(deposit ÷ reference_median
        × 100, deposit이 있을 때만), min/max_deposit(만원), recent(최근 5건), months_queried."""
        try:
            htype = HousingType(housing_type)
        except ValueError:
            valid = ", ".join(t.value for t in HousingType)
            return json.dumps(
                {"error": f"housing_type은 다음 중 하나여야 합니다: {valid}"}, ensure_ascii=False
            )

        records = []
        errors: list[str] = []
        ymds = recent_deal_months(months=months)
        for ymd in ymds:
            try:
                records.extend(client.fetch(lawd_cd, ymd, housing_type=htype))
            except MolitApiError as e:
                errors.append(f"{ymd}: {e}")
        summary = summarize_jeonse(records, building_name=building_name, area_m2=area_m2)
        if summary.new_contract_count >= MIN_NEW_CONTRACTS:
            reference, basis = summary.new_contract_median, "신규 계약 중위값"
        else:
            reference, basis = summary.median_deposit, "전체(갱신 포함, 신규 3건 미만)"
        ratio = round(deposit / reference * 100, 1) if (deposit and reference) else None
        payload: dict = {
            "housing_type": htype.value,
            "count": summary.count,
            "median_deposit": summary.median_deposit,
            "min_deposit": summary.min_deposit,
            "max_deposit": summary.max_deposit,
            "new_contract_count": summary.new_contract_count,
            "new_contract_median": summary.new_contract_median,
            "reference_median": reference,
            "reference_basis": basis if reference is not None else None,
            "ratio_to_reference": ratio,
            "recent": [
                {
                    **asdict(r),
                    "housing_type": r.housing_type.value,
                    "deal_date": r.deal_date.isoformat(),
                }
                for r in summary.recent
            ],
            "months_queried": ymds,
        }
        if errors:
            payload["errors"] = errors
        if summary.count == 0:
            payload["message"] = (
                "조건에 맞는 순수 전세 거래가 없습니다. "
                "건물명 표기, 면적, 조회 기간, 주거 유형을 바꿔 보세요."
            )
        return json.dumps(payload, ensure_ascii=False)

    return find_region_code, get_recent_jeonse_deals


def get_rent_client(settings: Settings) -> RentClient:
    if settings.molit_use_mock:
        return MockMolitRentClient()
    endpoints = {
        HousingType.APARTMENT: settings.apartment_openapi_endpoint,
        HousingType.MULTI_HOUSE: settings.multi_house_openapi_endpoint,
        HousingType.OFFICETEL: settings.office_openapi_endpoint,
    }
    return MolitRentClient(endpoints, settings.apartment_openapi_key_decoded)


def build_market_agent(settings: Settings, client: RentClient | None = None):
    tools = make_market_tools(client or get_rent_client(settings))
    return create_agent(
        model=get_llm(settings),
        tools=list(tools),
        system_prompt=MARKET_PROMPT,
        name="market_agent",
    )
