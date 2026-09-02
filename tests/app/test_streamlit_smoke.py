"""streamlit.testing.v1.AppTest: 실제 브라우저·LLM 없이 스크립트를 실행해 위젯 트리를 검사한다.
build_graph를 가짜 그래프로 바꿔 끼워 UI 배선만 검증한다.

바꿔 끼우는 지점이 `rent_agent.agents.supervisor.build_graph`인 이유:
AppTest는 streamlit_app.py를 `rent_agent.app.streamlit_app` 모듈이 아니라 독립 스크립트로
exec하므로, 그 모듈 객체의 속성을 바꿔도 스크립트가 만든 전역에는 영향이 없다.
앱이 `supervisor.build_graph(...)`처럼 모듈 속성으로 호출하면 호출 시점에 해석되어 통한다.
"""

from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from streamlit.testing.v1 import AppTest

# AppTest는 상대 경로를 호출한 파일 기준으로 푼다 → 레포 루트에서 절대 경로로 만든다.
APP = str(Path(__file__).resolve().parents[2] / "src" / "rent_agent" / "app" / "streamlit_app.py")


def _risk_turn(prompt: str, **final_kwargs) -> list:
    """한 턴: supervisor → risk_agent 핸드오프 + 위험 판단 도구 + 최종 답."""
    return [
        HumanMessage(prompt),
        AIMessage(
            "",
            name="supervisor",
            tool_calls=[{"name": "transfer_to_risk_agent", "args": {}, "id": "c1"}],
        ),
        ToolMessage("ok", tool_call_id="c1", name="transfer_to_risk_agent"),
        AIMessage(
            "",
            name="risk_agent",
            tool_calls=[{"name": "assess_jeonse_risk", "args": {}, "id": "c2"}],
        ),
        ToolMessage("{}", tool_call_id="c2", name="assess_jeonse_risk"),
        AIMessage("## 종합 판정: 위험\n테스트 리포트", name="supervisor", **final_kwargs),
    ]


def _earlier_knowledge_turn() -> list:
    """이전 턴(체크포인터에 누적된 히스토리). 실행 흐름에 나오면 안 된다."""
    return [
        HumanMessage("이전 질문"),
        AIMessage(
            "",
            name="supervisor",
            tool_calls=[{"name": "transfer_to_knowledge_agent", "args": {}, "id": "p1"}],
        ),
        ToolMessage("ok", tool_call_id="p1", name="transfer_to_knowledge_agent"),
        AIMessage("이전 답", name="supervisor"),
    ]


class FakeGraph:
    def __init__(self, history=None, error: Exception | None = None):
        self.calls: list[str] = []
        # history: prompt를 받아 messages를 만드는 함수. 기본은 현재 턴만 담긴 위험 진단 흐름.
        self._history = history or _risk_turn
        self._error = error

    def invoke(self, inputs, config=None):
        prompt = inputs["messages"][0].content
        self.calls.append(prompt)
        if self._error is not None:
            raise self._error
        return {"messages": self._history(prompt)}


def _app(monkeypatch, fake: FakeGraph | None = None) -> tuple[AppTest, FakeGraph]:
    fake = fake or FakeGraph()
    from rent_agent.agents import supervisor

    monkeypatch.setattr(supervisor, "build_graph", lambda *a, **kw: fake)
    # @st.cache_resource는 프로세스 전역이라 테스트 간 가짜 그래프가 새어 나간다.
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30)
    return at, fake


def _trace_text(at: AppTest) -> str:
    """expander의 repr에는 자식 텍스트가 없으므로 자식 markdown 값을 모아 본다."""
    return "\n".join(m.value for e in at.expander for m in e.markdown)


def _fill_valid_diag(at: AppTest) -> None:
    at.selectbox(key="housing").select("연립·다세대(빌라)")
    at.number_input(key="deposit").set_value(45000)
    at.number_input(key="price").set_value(60000)


def test_page_renders_two_tabs_and_form(monkeypatch):
    at, _ = _app(monkeypatch)
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["💬 지식 Q&A", "🔎 전세 진단"]
    assert at.selectbox(key="housing").options == ["아파트", "연립·다세대(빌라)", "오피스텔"]
    assert at.button[0].label == "진단하기"


def test_diagnosis_form_builds_prompt_with_housing_type_and_shows_report(monkeypatch):
    at, fake = _app(monkeypatch)
    at.run()
    _fill_valid_diag(at)
    at.button[0].click().run()
    assert not at.exception
    assert len(fake.calls) == 1
    prompt = fake.calls[0]
    assert "multi_house" in prompt and "45000만원" in prompt and "60000만원" in prompt
    assert "  " not in prompt  # filter(None, ...) 덕분에 이중 공백이 없다
    assert any("## 종합 판정: 위험" in m.value for m in at.markdown)
    assert "risk_agent → assess_jeonse_risk" in _trace_text(at)


def test_diagnosis_form_requires_deposit_and_price(monkeypatch):
    at, fake = _app(monkeypatch)
    at.run()
    at.number_input(key="deposit").set_value(0)
    at.button[0].click().run()
    assert fake.calls == []
    assert at.error and "필수" in at.error[0].value


def test_trace_shows_only_current_turn_and_forwarded_marker(monkeypatch):
    def history(prompt: str) -> list:
        return _earlier_knowledge_turn() + _risk_turn(
            prompt, response_metadata={"forwarded_from": "report_agent"}
        )

    at, _ = _app(monkeypatch, FakeGraph(history=history))
    at.run()
    _fill_valid_diag(at)
    at.button[0].click().run()
    assert not at.exception
    trace = _trace_text(at)
    assert "risk_agent → assess_jeonse_risk" in trace
    assert "📎 report_agent 원문 그대로 전달" in trace
    # 이전 턴의 핸드오프는 이번 턴 흐름이 아니다.
    assert "transfer_to_knowledge_agent" not in trace


def test_diagnosis_result_persists_across_rerun(monkeypatch):
    at, fake = _app(monkeypatch)
    at.run()
    _fill_valid_diag(at)
    at.button[0].click().run()
    at.run()  # 다른 위젯 조작 없이 rerun (submitted=False)
    assert not at.exception
    assert len(fake.calls) == 1  # 재실행이 그래프를 다시 호출하지 않는다
    assert any("## 종합 판정: 위험" in m.value for m in at.markdown)


def test_graph_error_is_shown_as_st_error(monkeypatch):
    at, _ = _app(monkeypatch, FakeGraph(error=RuntimeError("quota")))
    at.run()
    _fill_valid_diag(at)
    at.button[0].click().run()
    assert not at.exception  # 트레이스백이 아니라 안내 메시지로 처리
    assert "실패" in at.error[0].value and "quota" in at.error[0].value
