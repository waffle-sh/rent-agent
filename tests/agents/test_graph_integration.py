"""실제 LLM을 호출한다. 실행: uv run pytest -m integration -s"""

import os

import pytest
from langchain_core.messages import HumanMessage

from rent_agent.agents.llm import configure_tracing
from rent_agent.agents.supervisor import build_graph
from rent_agent.config import PROJECT_ROOT, Settings

pytestmark = pytest.mark.integration


@pytest.fixture
def real_settings():
    # conftest는 .env를 차단하므로 통합 테스트만 실제 .env를 명시적으로 읽는다
    # (integration 마커 → 더미 미주입)
    s = Settings(_env_file=PROJECT_ROOT / ".env")
    configure_tracing(s)
    if not s.openai_api_key or s.openai_api_key.startswith("sk-test"):
        pytest.skip("OPENAI_API_KEY 필요")
    if not os.path.exists(s.chroma_dir):
        pytest.skip("먼저 scripts/ingest.py 실행")
    return s


def _agents_called(result) -> set[str]:
    return {m.name for m in result["messages"] if getattr(m, "name", None)}


def test_knowledge_question_routes_to_knowledge_agent(real_settings):
    graph = build_graph(real_settings)
    result = graph.invoke({"messages": [HumanMessage("전입신고하면 대항력은 언제부터 생기나요?")]})
    assert "knowledge_agent" in _agents_called(result)
    assert "risk_agent" not in _agents_called(result)
    assert "다음 날" in result["messages"][-1].content or "익일" in result["messages"][-1].content


def test_jeonse_diagnosis_calls_risk_and_report(real_settings):
    graph = build_graph(real_settings)
    prompt = (
        "서울 강남구 까치마을 39.6㎡ 전세를 보려고 합니다. 보증금 4억 5천, 매매 시세 6억, "
        "근저당 채권최고액 1억 2천, 자기자금 2억, 연소득 4천만원입니다. 괜찮은 계약인가요?"
    )
    result = graph.invoke({"messages": [HumanMessage(prompt)]})
    called = _agents_called(result)
    assert {"risk_agent", "report_agent"} <= called
    final = result["messages"][-1].content
    # forward_message로 리포트 원문이 그대로 전달됨
    assert final.lstrip().startswith("## 종합 판정")
    assert "위험" in final  # 전세가율 75%, 총 부담률 95% → 위험
    assert "법률·금융 자문이 아닙니다" in final
