"""Supervisor 그래프 조립.

- output_mode='full_history': report_agent가 다른 에이전트의 도구 결과(수치)를 직접 봐야 하고,
  UI에서 에이전트 호출 흐름을 그대로 보여 주기 위함.
- forward_message 도구: 통합 테스트에서 supervisor가 report_agent의 리포트를 자기 말로 바꿔 쓰며
  "## 종합 판정" 헤더를 유실하는 것이 관측됨(2026-09-02). 워커 메시지를 원문 그대로 사용자에게
  전달하는 langgraph-supervisor 내장 도구로 해결한다.
  프롬프트 지시만으로는 재작성 습관을 막지 못했다.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_forward_message_tool

from rent_agent.agents.knowledge_agent import build_knowledge_agent
from rent_agent.agents.llm import get_llm
from rent_agent.agents.market_agent import build_market_agent
from rent_agent.agents.prompts import SUPERVISOR_PROMPT
from rent_agent.agents.report_agent import build_report_agent
from rent_agent.agents.risk_agent import build_risk_agent
from rent_agent.config import Settings


def build_graph(settings: Settings, checkpointer: BaseCheckpointSaver | None = None):
    agents = [
        build_knowledge_agent(settings),
        build_market_agent(settings),
        build_risk_agent(settings),
        build_report_agent(settings),
    ]
    workflow = create_supervisor(
        agents,
        model=get_llm(settings),
        tools=[create_forward_message_tool("supervisor")],
        prompt=SUPERVISOR_PROMPT,
        output_mode="full_history",
        add_handoff_back_messages=True,
        supervisor_name="supervisor",
    )
    return workflow.compile(checkpointer=checkpointer)
