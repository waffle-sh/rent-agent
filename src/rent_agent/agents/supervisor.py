"""Supervisor 그래프 조립.

- output_mode='full_history': report_agent가 다른 에이전트의 도구 결과(수치)를 직접 봐야 하고,
  UI에서 에이전트 호출 흐름을 그대로 보여 주기 위함.
- 두 개의 결정적 후처리 (2026-09-02 통합 테스트 실측에 근거):
  1) ensure_report: supervisor가 risk_agent 결과를 받은 뒤 report_agent를 건너뛰고 직접 답한 경우가
     관측됨(2회 중 1회). 이번 턴에 assess_jeonse_risk가 실행됐는데 report_agent의 답이 없으면
     그래프가 report_agent를 직접 실행한다.
  2) preserve_worker_answer: supervisor가 report/knowledge 에이전트의 답을 재작성해 "## 종합 판정"
     헤더·면책 문구를 유실한 경우가 관측됨(3회 중 1회).
     forward_message 도구도 모델이 호출하지 않았다.
     마지막 사용자 턴 이후 워커의 최종 답이 있으면 supervisor의 마지막 메시지를 원문으로 교체한다.
  프롬프트 지시는 확률적이므로, 결과의 완결성·충실성은 LLM 판단에 맡기지 않고 그래프가 보장한다.
"""

from collections.abc import Callable
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph_supervisor import create_supervisor

from rent_agent.agents.knowledge_agent import build_knowledge_agent
from rent_agent.agents.llm import get_llm
from rent_agent.agents.market_agent import build_market_agent
from rent_agent.agents.prompts import SUPERVISOR_PROMPT
from rent_agent.agents.report_agent import build_report_agent
from rent_agent.agents.risk_agent import build_risk_agent
from rent_agent.config import Settings

SUPERVISOR_NAME = "supervisor"
REPORT_AGENT = "report_agent"
RISK_TOOL = "assess_jeonse_risk"
# risk_agent 도구의 검증 실패 접두어 (risk_agent.py와 일치해야 함)
TOOL_ERROR_PREFIX = "입력 오류"
# 이 에이전트들의 답은 사용자에게 원문 그대로 가야 한다 (수치·근거 URL·면책 문구 보존)
VERBATIM_AGENTS = (REPORT_AGENT, "knowledge_agent")


def _current_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    """마지막 HumanMessage 이후 구간 (멀티턴에서 이전 턴의 결과를 끌어오지 않도록)."""
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    return messages[last_human + 1 :]


def _last_worker_answer(turn: list[BaseMessage], names: tuple[str, ...]) -> AIMessage | None:
    """도구 호출이 달린 AIMessage(핸드오프)는 워커의 '답'이 아니므로 제외."""
    return next(
        (
            m
            for m in reversed(turn)
            if isinstance(m, AIMessage) and m.name in names and not m.tool_calls and m.content
        ),
        None,
    )


def needs_report(state: MessagesState) -> Literal["report", "preserve"]:
    """이번 턴에 위험 판단 도구가 **유효한 결과**를 냈는데 report_agent의 답이 없으면
    리포트 단계로 보낸다.
    도구가 "입력 오류"를 돌려준 경우(추출 실패 → supervisor가 되묻는 게 맞음)는
    리포트를 강제하지 않는다."""
    turn = _current_turn(state["messages"])
    ran_risk = any(
        isinstance(m, ToolMessage)
        and m.name == RISK_TOOL
        and not str(m.content).startswith(TOOL_ERROR_PREFIX)
        for m in turn
    )
    has_report = _last_worker_answer(turn, (REPORT_AGENT,)) is not None
    return "report" if ran_risk and not has_report else "preserve"


def make_report_node(report_agent) -> Callable[[MessagesState], dict]:
    """report_agent를 단독 실행한다. create_agent(name=...)가 AIMessage.name을 붙이지만,
    후처리 노드들이 이름에 의존하므로 방어적으로 한 번 더 보장한다."""

    def run_report(state: MessagesState) -> dict:
        before = len(state["messages"])
        result = report_agent.invoke({"messages": state["messages"]})
        new_messages = []
        for m in result["messages"][before:]:
            if isinstance(m, AIMessage):
                m = m.model_copy(update={"name": REPORT_AGENT})
            new_messages.append(m)
        return {"messages": new_messages}

    return run_report


def preserve_worker_answer(state: MessagesState) -> dict:
    """supervisor의 마지막 답이 워커(report/knowledge)의 최종 답을 재작성한 것이면
    원문으로 교체한다.

    - 워커 답이 없거나(예: supervisor가 되묻는 경우) 이미 동일하면 아무것도 바꾸지 않는다.
    - report_agent 답은 항상 원문 우선(종합 리포트가 곧 최종 답).
    - knowledge_agent 답은 **이 턴에서 답한 워커가 그것 하나일 때만** 교체한다.
      시세+지식처럼 여러 워커가 답한 턴에서는 supervisor의 종합이 정당하며,
      지식 답만으로 덮으면 시세 결과가 사라진다.
    """
    messages = state["messages"]
    if not messages:
        return {}
    turn = _current_turn(messages)
    worker = _last_worker_answer(turn, VERBATIM_AGENTS)
    final = messages[-1]
    if worker is None or not isinstance(final, AIMessage) or final.name != SUPERVISOR_NAME:
        return {}
    if worker.name != REPORT_AGENT:
        answered = {
            m.name
            for m in turn
            if isinstance(m, AIMessage)
            and m.name != SUPERVISOR_NAME
            and not m.tool_calls
            and m.content
        }
        if len(answered) > 1:
            return {}
    if str(final.content).strip() == str(worker.content).strip():
        return {}
    # 같은 id로 돌려주면 add_messages 리듀서가 기존 메시지를 교체한다. usage_metadata 등은 유지.
    return {
        "messages": [
            final.model_copy(
                update={
                    "content": worker.content,
                    "response_metadata": {
                        **final.response_metadata,
                        "forwarded_from": worker.name,
                    },
                }
            )
        ]
    }


def build_graph(settings: Settings, checkpointer: BaseCheckpointSaver | None = None):
    report_agent = build_report_agent(settings)
    agents = [
        build_knowledge_agent(settings),
        build_market_agent(settings),
        build_risk_agent(settings),
        report_agent,
    ]
    team = create_supervisor(
        agents,
        model=get_llm(settings),
        prompt=SUPERVISOR_PROMPT,
        output_mode="full_history",
        add_handoff_back_messages=True,
        supervisor_name=SUPERVISOR_NAME,
    ).compile()

    outer = StateGraph(MessagesState)
    outer.add_node("team", team)
    outer.add_node("report", make_report_node(report_agent))
    outer.add_node("preserve_worker_answer", preserve_worker_answer)
    outer.add_edge(START, "team")
    outer.add_conditional_edges(
        "team", needs_report, {"report": "report", "preserve": "preserve_worker_answer"}
    )
    outer.add_edge("report", "preserve_worker_answer")
    outer.add_edge("preserve_worker_answer", END)
    return outer.compile(checkpointer=checkpointer)
