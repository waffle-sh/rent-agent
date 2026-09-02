"""지식 QA 에이전트: Chroma 검색 도구 + 근거 인용 프롬프트."""

from langchain.agents import create_agent
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool

from rent_agent.agents.llm import get_llm
from rent_agent.agents.prompts import KNOWLEDGE_PROMPT
from rent_agent.config import Settings
from rent_agent.rag.retriever import get_retriever


def make_knowledge_tool(retriever: BaseRetriever) -> BaseTool:
    @tool
    def search_real_estate_knowledge(query: str) -> str:
        """부동산 임대차 법령(주택임대차보호법), 소액임차인 최우선변제 기준,
        전세사기 예방 체크리스트,
        HUG 전세보증금반환보증, 청년·버팀목 전세대출 자료를 검색한다.
        질문을 그대로 넣으면 관련 문단과 출처를 반환한다."""
        docs = retriever.invoke(query)
        if not docs:
            return "관련 문서를 찾지 못했습니다."
        return "\n\n---\n\n".join(
            f"[출처: {d.metadata.get('title', '')} | {d.metadata.get('source', '')} | "
            f"기준일 {d.metadata.get('effective_date', '')}]\n{d.page_content}"
            for d in docs
        )

    return search_real_estate_knowledge


def build_knowledge_agent(settings: Settings, retriever: BaseRetriever | None = None):
    search = make_knowledge_tool(retriever or get_retriever(settings))
    return create_agent(
        model=get_llm(settings),
        tools=[search],
        system_prompt=KNOWLEDGE_PROMPT,
        name="knowledge_agent",
    )
