"""리트리버 팩토리. 단순 유사도 top-k를 쓰는 이유: 코퍼스가 섹션당 1청크인 비중복 구조라
MMR의 다양성 항은 무관한 문서만 끌어온다(실측: 4개 질의 중 3개에서 관련 청크가 밀림).
계획 Task 8 설계 근거 참고."""

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from rent_agent.config import Settings
from rent_agent.rag.ingest import get_embedding


def get_vectorstore(settings: Settings, embedding: Embeddings | None = None) -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embedding or get_embedding(settings),
        persist_directory=str(settings.chroma_dir),
    )


def get_retriever(settings: Settings, embedding: Embeddings | None = None) -> VectorStoreRetriever:
    return get_vectorstore(settings, embedding).as_retriever(
        search_type="similarity", search_kwargs={"k": settings.retriever_k}
    )
