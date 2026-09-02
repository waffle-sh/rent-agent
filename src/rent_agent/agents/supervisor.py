"""Supervisor 그래프 조립.

- output_mode='full_history': report_agent가 다른 에이전트의 도구 결과(수치)를 직접 봐야 하고,
  UI에서 에이전트 호출 흐름을 그대로 보여 주기 위함.
- preserve_worker_answer 노드: 통합 테스트에서 supervisor가 report_agent의 리포트를 자기 말로
  바꿔 쓰며 "## 종합 판정" 헤더·면책 문구를 유실하는 것이 관측됨(2026-09-02, 3회 중 1회).
  프롬프트 지시와 langgraph-supervisor의 forward_message 도구 모두 모델이 따르지 않을 수 있어
  (도구 미호출 관측),
  LLM 판단에 맡기지 않고 **결정적 후처리**로 보장한다: 마지막 사용자 메시지 이후 report_agent /
  knowledge_agent의 최종 답이 있으면 supervisor의 마지막 메시지를 그 원문으로 교체한다.
"""

from langchain_core.messages import AIMessage, HumanMessage
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
# 이 에이전트들의 답은 사용자에게 원문 그대로 가야 한다 (수치·근거 URL·면책 문구 보존)
VERBATIM_AGENTS = ("report_agent", "knowledge_agent")


def preserve_worker_answer(state: MessagesState) -> dict:
    """supervisor의 마지막 답이 워커(report/knowledge)의 최종 답을 재작성한 것이면
    원문으로 교체한다.

    - 마지막 HumanMessage 이후 구간만 본다 (멀티턴 대화에서 이전 턴의 리포트를 끌어오지 않도록).
    - 도구 호출이 달린 AIMessage(핸드오프)는 워커의 '답'이 아니므로 제외한다.
    - 워커 답이 없거나(예: supervisor가 되묻는 경우) 이미 동일하면 아무것도 바꾸지 않는다.
    """
    messages = state["messages"]
    if not messages:
        return {}
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    tail = messages[last_human + 1 :]
    worker = next(
        (
            m
            for m in reversed(tail)
            if isinstance(m, AIMessage)
            and m.name in VERBATIM_AGENTS
            and not m.tool_calls
            and m.content
        ),
        None,
    )
    final = messages[-1]
    if worker is None or not isinstance(final, AIMessage) or final.name != SUPERVISOR_NAME:
        return {}
    if str(final.content).strip() == str(worker.content).strip():
        return {}
    # 같은 id로 돌려주면 add_messages 리듀서가 기존 메시지를 교체한다
    return {
        "messages": [
            AIMessage(
                id=final.id,
                content=worker.content,
                name=SUPERVISOR_NAME,
                response_metadata={"forwarded_from": worker.name},
            )
        ]
    }


def build_graph(settings: Settings, checkpointer: BaseCheckpointSaver | None = None):
    agents = [
        build_knowledge_agent(settings),
        build_market_agent(settings),
        build_risk_agent(settings),
        build_report_agent(settings),
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
    outer.add_node("preserve_worker_answer", preserve_worker_answer)
    outer.add_edge(START, "team")
    outer.add_edge("team", "preserve_worker_answer")
    outer.add_edge("preserve_worker_answer", END)
    return outer.compile(checkpointer=checkpointer)
