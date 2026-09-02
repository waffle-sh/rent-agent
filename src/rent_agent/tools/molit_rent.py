"""국토교통부 전월세 실거래가 API 클라이언트 (아파트 / 연립·다세대 / 오피스텔).

- 문서: 공공데이터포털 "국토교통부_{아파트|연립다세대|오피스텔} 전월세 실거래가 자료" (docs/*.hwp)
- 요청: GET {endpoint}/{operation}?serviceKey&LAWD_CD(5자리)&DEAL_YMD(YYYYMM)&pageNo&numOfRows
- 응답: XML. 세 API의 골격은 같고 건물명 필드만 다르다.
  금액은 '만원' 단위 문자열에 콤마 포함 ("100,000").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import httpx
import xmltodict

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures"


class HousingType(StrEnum):
    APARTMENT = "apartment"
    MULTI_HOUSE = "multi_house"  # 연립·다세대 (빌라)
    OFFICETEL = "officetel"


@dataclass(frozen=True)
class HousingSpec:
    operation: str  # API 오퍼레이션 이름
    name_field: str  # 건물명 XML 필드
    fixture: str  # Mock용 픽스처 파일명


HOUSING_SPECS: dict[HousingType, HousingSpec] = {
    HousingType.APARTMENT: HousingSpec(
        "getRTMSDataSvcAptRent", "aptNm", "rent_response.xml"
    ),
    HousingType.MULTI_HOUSE: HousingSpec(
        "getRTMSDataSvcRHRent", "mhouseNm", "rent_response_rh.xml"
    ),
    HousingType.OFFICETEL: HousingSpec(
        "getRTMSDataSvcOffiRent", "offiNm", "rent_response_offi.xml"
    ),
}


class MolitApiError(RuntimeError):
    """API 오류(키 미등록, 서비스 없음, 쿼터 초과, HTTP 오류 등)."""


@dataclass(frozen=True)
class RentRecord:
    housing_type: HousingType
    building_name: str
    sub_type: str  # 연립·다세대만 "연립" | "다세대", 그 외 ""
    dong: str
    area_m2: float
    floor: int
    build_year: int
    deal_date: date
    deposit: int  # 만원
    monthly_rent: int  # 만원, 0이면 전세
    contract_type: str  # "신규" | "갱신" | ""
    renewal_right_used: bool

    @property
    def is_jeonse(self) -> bool:
        return self.monthly_rent == 0


def _to_int(value: str | None) -> int:
    if value is None:
        return 0
    cleaned = value.replace(",", "").strip()
    return int(cleaned) if cleaned else 0


def _to_float(value: str | None) -> float:
    return float(value.strip()) if value and value.strip() else 0.0


def _clean(value: str | None) -> str:
    return (value or "").strip()


def parse_rent_xml(xml_text: str, housing_type: HousingType) -> tuple[list[RentRecord], int]:
    """XML → (레코드 목록, totalCount). 공공데이터포털 공통 에러 포맷은 예외로 변환."""
    data = xmltodict.parse(xml_text)
    if "OpenAPI_ServiceResponse" in data:
        header = data["OpenAPI_ServiceResponse"].get("cmmMsgHeader", {})
        raise MolitApiError(f"{header.get('errMsg')}: {header.get('returnAuthMsg')}")

    response = data["response"]
    result_code = _clean(response.get("header", {}).get("resultCode"))
    if result_code not in ("000", "00"):
        result_msg = response.get("header", {}).get("resultMsg")
        raise MolitApiError(f"resultCode={result_code}: {result_msg}")

    body = response.get("body") or {}
    total = _to_int(body.get("totalCount"))
    items = (body.get("items") or {}).get("item") or []
    if isinstance(items, dict):  # 결과가 1건이면 dict로 옴
        items = [items]

    name_field = HOUSING_SPECS[housing_type].name_field
    records = [
        RentRecord(
            housing_type=housing_type,
            building_name=_clean(it.get(name_field)),
            sub_type=_clean(it.get("houseType")),
            dong=_clean(it.get("umdNm")),
            area_m2=_to_float(it.get("excluUseAr")),
            floor=_to_int(it.get("floor")),
            build_year=_to_int(it.get("buildYear")),
            deal_date=date(
                _to_int(it.get("dealYear")),
                _to_int(it.get("dealMonth")),
                _to_int(it.get("dealDay")),
            ),
            deposit=_to_int(it.get("deposit")),
            monthly_rent=_to_int(it.get("monthlyRent")),
            contract_type=_clean(it.get("contractType")),
            renewal_right_used=_clean(it.get("useRRRight")) == "사용",
        )
        for it in items
    ]
    return records, total


class RentClient(Protocol):
    def fetch(
        self,
        lawd_cd: str,
        deal_ymd: str,
        housing_type: HousingType = HousingType.APARTMENT,
        num_of_rows: int = 1000,
    ) -> list[RentRecord]: ...


class MolitRentClient:
    def __init__(
        self, endpoints: dict[HousingType, str], service_key: str, http: httpx.Client | None = None
    ) -> None:
        self._endpoints = {k: v.rstrip("/") for k, v in endpoints.items()}
        self._key = service_key  # 디코딩된 키. httpx가 params로 한 번만 인코딩한다.
        self._http = http or httpx.Client(timeout=15.0)

    def fetch(
        self,
        lawd_cd: str,
        deal_ymd: str,
        housing_type: HousingType = HousingType.APARTMENT,
        num_of_rows: int = 1000,
    ) -> list[RentRecord]:
        endpoint = self._endpoints.get(housing_type)
        if not endpoint:
            raise MolitApiError(f"{housing_type.value} 유형의 엔드포인트가 설정되지 않았습니다")
        params = {
            "serviceKey": self._key,
            "LAWD_CD": lawd_cd,
            "DEAL_YMD": deal_ymd,
            "pageNo": 1,
            "numOfRows": num_of_rows,
        }
        try:
            resp = self._http.get(
                f"{endpoint}/{HOUSING_SPECS[housing_type].operation}", params=params
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise MolitApiError(f"HTTP 오류: {e}") from e
        records, _ = parse_rent_xml(resp.text, housing_type)
        return records


class MockMolitRentClient:
    """키 없이 개발/테스트용. 유형별 픽스처 XML을 그대로 반환한다."""

    def fetch(
        self,
        lawd_cd: str,
        deal_ymd: str,
        housing_type: HousingType = HousingType.APARTMENT,
        num_of_rows: int = 1000,
    ) -> list[RentRecord]:
        xml_text = (FIXTURE_DIR / HOUSING_SPECS[housing_type].fixture).read_text(encoding="utf-8")
        records, _ = parse_rent_xml(xml_text, housing_type)
        return records
