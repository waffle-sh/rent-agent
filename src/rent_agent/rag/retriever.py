"""Chroma 벡터스토어/리트리버 팩토리."""

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
    # MMR: 같은 조문의 인접 청크만 잔뜩 뽑히는 것을 막고 서로 다른 근거를 섞어 준다.
    return get_vectorstore(settings, embedding).as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.retriever_k, "fetch_k": settings.retriever_k * 4},
    )
