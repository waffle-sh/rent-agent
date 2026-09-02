"""청킹 + Chroma 적재. 임베딩을 주입받아 테스트에서는 Fake, 운영에서는 OpenAI를 쓴다."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rent_agent.config import Settings
from rent_agent.rag.loader import load_markdown_docs

# 마크다운 헤더 → 빈 줄 → 줄 → 문장 순으로 자른다. 조문/항목 경계를 우선 존중.
SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]


def split_documents(
    docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 100
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=SEPARATORS
    )
    chunks = splitter.split_documents(docs)
    # 청크 앞에 "[문서 제목] "을 붙인다. "## 보증금액 한도"처럼 맥락이 약한 헤더 청크도
    # 어떤 문서(HUG 보증/버팀목 대출 등)의 내용인지 임베딩과 LLM이 알 수 있게 하기 위함.
    for c in chunks:
        c.page_content = f"[{c.metadata.get('title', '')}] {c.page_content}"
    return chunks


def build_vectorstore(
    raw_dir: Path, chroma_dir: Path, embedding: Embeddings, collection: str, reset: bool = False
) -> Chroma:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    if reset:
        # 계획의 shutil.rmtree 대신 컬렉션만 비운다(Step 5 대안).
        # 같은 경로에 이미 열린 chromadb 클라이언트가 있는 상태에서 디렉터리를 지우면
        # "attempt to write a readonly database"로 실패한다.
        existing = Chroma(
            collection_name=collection,
            embedding_function=embedding,
            persist_directory=str(chroma_dir),
        )
        existing.reset_collection()
    chunks = split_documents(load_markdown_docs(raw_dir))
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=collection,
        persist_directory=str(chroma_dir),
    )


def get_embedding(settings: Settings) -> Embeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model, api_key=settings.openai_api_key
    )


def ingest(settings: Settings, reset: bool = True) -> int:
    vs = build_vectorstore(
        raw_dir=settings.raw_docs_dir,
        chroma_dir=settings.chroma_dir,
        embedding=get_embedding(settings),
        collection=settings.chroma_collection,
        reset=reset,
    )
    return vs._collection.count()
