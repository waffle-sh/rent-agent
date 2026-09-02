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
    assert isinstance(data["reasons"], list) and data["reasons"]


def test_tool_invalid_region_message():
    out = assess_jeonse_risk.invoke({"deposit": 1000, "market_price": 5000, "region": "mars"})
    assert "region" in out and "seoul" in out


def test_tool_has_korean_description():
    assert "보증금" in assess_jeonse_risk.description
