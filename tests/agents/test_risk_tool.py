import json

from rent_agent.agents.risk_agent import assess_jeonse_risk


def test_tool_returns_json_assessment():
    out = assess_jeonse_risk.invoke(
        {
            "deposit": 35000,
            "market_price": 50000,
            "senior_liens": 10000,
            "region": "seoul",
            "own_capital": 15000,
        }
    )
    data = json.loads(out)
    assert data["level"] == "위험"
    assert data["jeonse_ratio"] == 70.0
    assert data["required_loan"] == 20000
    assert isinstance(data["reasons"], list) and any("전세가율" in r for r in data["reasons"])


def test_tool_safe_case_with_income():
    out = assess_jeonse_risk.invoke(
        {
            "deposit": 20000,
            "market_price": 50000,
            "own_capital": 5000,
            "annual_income": 6000,
            "loan_rate": 4.0,
        }
    )
    data = json.loads(out)
    assert data["level"] == "안전"
    assert data["required_loan"] == 15000
    assert data["interest_to_income_ratio"] == 10.0  # 월이자 50만 / 월소득 500만


def test_region_is_an_enum_in_tool_schema():
    # LLM이 허용값을 스키마에서 바로 보도록 Literal 사용 (설명 텍스트에만 의존하지 않음)
    props = assess_jeonse_risk.args_schema.model_json_schema()["properties"]
    assert props["region"]["enum"] == ["seoul", "metro_over", "metro_city", "other"]
    assert props["region"]["default"] == "seoul"


def test_validation_error_is_compact_korean():
    out = assess_jeonse_risk.invoke({"deposit": 1000, "market_price": 0})
    assert out.startswith("입력 오류: market_price: 0보다 커야 합니다")
    assert "pydantic" not in out and "{" not in out


def test_won_instead_of_manwon_is_rejected():
    # 3.5억을 원 단위(350,000,000)로 넣으면 계산은 되지만 결과가 무의미 → 단위 확인 요구
    out = assess_jeonse_risk.invoke({"deposit": 350_000_000, "market_price": 500_000_000})
    assert "만원 단위" in out and "입력 오류" in out


def test_tool_has_korean_description():
    assert "보증금" in assess_jeonse_risk.description and "만원" in assess_jeonse_risk.description
