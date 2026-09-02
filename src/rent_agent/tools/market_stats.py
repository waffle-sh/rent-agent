"""실거래 레코드 → 전세 시세 요약. 중위값을 쓰는 이유: 소수 고가/저가 거래에 덜 민감."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from rent_agent.tools.molit_rent import RentRecord


@dataclass(frozen=True)
class JeonseMarketSummary:
    count: int
    median_deposit: int | None
    min_deposit: int | None
    max_deposit: int | None
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
    """순수 전세(월세 0)만 대상으로 건물명 부분일치·전용면적 ±허용치로 필터 후 요약.
    주거 유형은 호출자가 이미 분리해 넘긴다."""
    filtered = [
        r
        for r in records
        if r.is_jeonse
        and (building_name is None or building_name in r.building_name)
        and (area_m2 is None or abs(r.area_m2 - area_m2) <= area_tolerance)
    ]
    if not filtered:
        return JeonseMarketSummary(count=0, median_deposit=None, min_deposit=None, max_deposit=None)

    deposits = [r.deposit for r in filtered]
    recent = sorted(filtered, key=lambda r: r.deal_date, reverse=True)[:5]
    return JeonseMarketSummary(
        count=len(filtered),
        median_deposit=int(median(deposits)),
        min_deposit=min(deposits),
        max_deposit=max(deposits),
        recent=recent,
    )
