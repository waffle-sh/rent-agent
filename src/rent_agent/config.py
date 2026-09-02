"""환경 변수 기반 설정. .env 파일은 pydantic-settings가 읽는다."""

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 테스트가 실제 .env를 읽지 않도록 경로를 환경변수로 바꿔 끼울 수 있게 한다 (conftest 참고).
ENV_FILE = os.getenv("RENT_AGENT_ENV_FILE", str(PROJECT_ROOT / ".env"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # 국토부 전월세 실거래가 (공공데이터포털). 서비스 키 하나로 세 API 모두 호출한다.
    apartment_openapi_key: str
    apartment_openapi_endpoint: str = "https://apis.data.go.kr/1613000/RTMSDataSvcAptRent"
    # 연립·다세대
    multi_house_openapi_endpoint: str = "https://apis.data.go.kr/1613000/RTMSDataSvcRHRent"
    # 오피스텔
    office_openapi_endpoint: str = "https://apis.data.go.kr/1613000/RTMSDataSvcOffiRent"
    molit_use_mock: bool = False

    # RAG
    raw_docs_dir: Path = PROJECT_ROOT / "data" / "raw"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    chroma_collection: str = "real_estate_knowledge"
    # 53청크 코퍼스에서 상품 비교형 질의는 관련 섹션이 4~6개라
    # k=4는 정답 섹션을 밀어냄 (RAGAS Q9 실측). ADR-0002
    retriever_k: int = 6

    # LangSmith. 주의: pydantic-settings는 .env를 os.environ에 올리지 않는다.
    # 트레이서는 os.environ만 읽으므로 진입점에서 agents.llm.configure_tracing()을 호출해야 한다.
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "rent-agent"

    @property
    def apartment_openapi_key_decoded(self) -> str:
        """공공데이터포털은 'Encoding/Decoding' 두 키를 준다. 어떤 것을 넣어도 동작하도록
        항상 디코딩하고, HTTP 클라이언트가 한 번만 인코딩하게 한다."""
        return unquote(self.apartment_openapi_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
