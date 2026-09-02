import os

from langchain_openai import ChatOpenAI

from rent_agent.config import Settings


def get_llm(settings: Settings, temperature: float = 0.0) -> ChatOpenAI:
    # temperature 0: 라우팅·수치 설명은 재현성이 중요. 창의성 불필요.
    return ChatOpenAI(
        model=settings.openai_model, temperature=temperature, api_key=settings.openai_api_key
    )


def configure_tracing(settings: Settings) -> None:
    """LangSmith 트레이싱을 켠다. .env 값을 프로세스 환경변수로 올려 langchain 트레이서가 읽게 한다.
    이미 환경변수가 있으면 덮어쓰지 않는다."""
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        return
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
