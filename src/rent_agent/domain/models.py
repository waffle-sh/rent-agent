"""전세 위험 판단 입출력 모델. 금액 단위는 모두 '만원'."""

from enum import StrEnum

from pydantic import BaseModel, Field


class Region(StrEnum):
    """소액임차인 최우선변제 기준 지역 구분 (주택임대차보호법 시행령 제10조·제11조)."""

    SEOUL = "seoul"  # 서울특별시
    METRO_OVER = "metro_over"  # 과밀억제권역(서울 제외), 세종, 용인, 화성, 김포
    METRO_CITY = "metro_city"  # 광역시(과밀억제권역·군 제외), 안산, 광주, 파주, 이천, 평택
    OTHER = "other"  # 그 밖의 지역


class JeonseInput(BaseModel):
    deposit: int = Field(..., gt=0, description="전세 보증금 (만원)")
    market_price: int = Field(..., gt=0, description="해당 주택 매매 시세 (만원)")
    senior_liens: int = Field(
        0, ge=0, description="등기부 을구 선순위 근저당 채권최고액 합계 (만원)"
    )
    senior_deposits: int = Field(
        0, ge=0, description="나보다 먼저 들어온 임차인 보증금 합계 (만원, 다가구 등)"
    )
    region: Region = Field(Region.SEOUL, description="소액임차인 기준 지역")
    own_capital: int = Field(0, ge=0, description="자기자금 (만원)")
    annual_income: int | None = Field(None, gt=0, description="연소득 (만원)")
    loan_rate: float = Field(3.5, ge=0, description="전세대출 예상 금리 (연 %)")
    auction_ratio: float = Field(0.8, gt=0, le=1, description="경매 낙찰가율 가정 (0~1)")


class RiskLevel(StrEnum):
    SAFE = "안전"
    CAUTION = "주의"
    DANGER = "위험"
    CRITICAL = "매우 위험"


class RiskAssessment(BaseModel):
    jeonse_ratio: float = Field(description="전세가율 (%)")
    total_burden_ratio: float = Field(description="선순위 포함 총 부담률 (%)")
    expected_recovery: int = Field(description="경매 시 예상 회수 가능액 (만원)")
    shortfall: int = Field(description="보증금 대비 회수 부족액 (만원, 0이면 전액 회수 가정)")
    small_tenant_protected: bool = Field(description="소액임차인 최우선변제 대상 여부")
    small_tenant_priority_amount: int = Field(description="최우선변제 가능액 (만원)")
    required_loan: int = Field(description="필요 대출액 (만원)")
    monthly_interest: float = Field(description="월 이자 (만원)")
    interest_to_income_ratio: float | None = Field(description="월 이자 / 월 소득 (%)")
    level: RiskLevel
    reasons: list[str] = Field(description="판정 근거 (사람이 읽는 문장)")
