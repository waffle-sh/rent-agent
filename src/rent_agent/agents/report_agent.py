"""리포트 에이전트: 도구 없음. 대화 내 다른 에이전트 결과만 종합한다."""

from langchain.agents import create_agent

from rent_agent.agents.llm import get_llm
from rent_agent.agents.prompts import REPORT_PROMPT
from rent_agent.config import Settings


def build_report_agent(settings: Settings):
    # 리포트는 약간의 문장 다양성이 읽기 좋아 temperature 0.3
    return create_agent(
        model=get_llm(settings, temperature=0.3),
        tools=[],
        system_prompt=REPORT_PROMPT,
        name="report_agent",
    )
