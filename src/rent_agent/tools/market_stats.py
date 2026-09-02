"""실거래 레코드 → 전세 시세 요약.

- 중위값을 쓰는 이유: 소수 고가/저가 거래에 덜 민감.
  (짝수 개면 두 중앙값 평균을 절사 — 만원 단위 오차 0.5 이하)
- 갱신 계약은 증액 상한 5% 때문에 2년 전 가격을 반영하므로, 신규 계약만의 중위값을 별도로 제공한다.
  계약구분이 비어 있는 행(2021년 이전 계약 등)은 신규로 간주한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from rent_agent.tools.molit_rent import RentRecord

RENEWAL = "갱신"


def _norm(text: str) -> str:
    """공백 제거 + 대소문자 무시.
    API 표기("강남 지웰홈스")와 사용자 입력("강남지웰홈스") 차이를 흡수."""
    return "".join(text.split()).casefold()


@dataclass(frozen=True)
class JeonseMarketSummary:
    count: int
    median_deposit: int | None
    min_deposit: int | None
    max_deposit: int | None
    new_contract_count: int = 0
    new_contract_median: int | None = None  # 갱신 제외 중위값. 시세 비교 시 우선 사용
    recent: list[RentRecord] = field(default_factory=list)  # 최신순, 최대 5건

    def ratio_to_median(self, deposit: int) -> float | None:
        if not self.median_deposit:
            return None
        return round(deposit / self.median_deposit * 100, 1)


def summarize_jeonse(
    records: list[RentRecord],
    building_name: str | None = None,
    area_m2: float | None = None,
    area_tolerance: float = 5.0,
) -> JeonseMarketSummary:
    """순수 전세(월세 0)만 대상으로 건물명 부분일치(정규화)·전용면적 ±허용치로 필터 후 요약.
    주거 유형은 호출자가 이미 분리해 넘긴다. building_name이 빈 문자열이면 필터하지 않는다."""
    name_key = _norm(building_name) if building_name else ""
    filtered = [
        r
        for r in records
        if r.is_jeonse
        and (not name_key or name_key in _norm(r.building_name))
        and (area_m2 is None or abs(r.area_m2 - area_m2) <= area_tolerance)
    ]
    if not filtered:
        return JeonseMarketSummary(count=0, median_deposit=None, min_deposit=None, max_deposit=None)

    deposits = [r.deposit for r in filtered]
    new_deposits = [r.deposit for r in filtered if r.contract_type != RENEWAL]
    # 같은 날 거래가 많으므로 (거래일, 보증금) 내림차순으로 결정적 정렬
    recent = sorted(filtered, key=lambda r: (r.deal_date, r.deposit), reverse=True)[:5]
    return JeonseMarketSummary(
        count=len(filtered),
        median_deposit=int(median(deposits)),
        min_deposit=min(deposits),
        max_deposit=max(deposits),
        new_contract_count=len(new_deposits),
        new_contract_median=int(median(new_deposits)) if new_deposits else None,
        recent=recent,
    )
