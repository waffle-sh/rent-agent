"""청킹 + Chroma 적재. 임베딩을 주입받아 테스트에서는 Fake, 운영에서는 OpenAI를 쓴다.

청킹 전략(근거는 계획 Task 8 "설계 근거"):
1) MarkdownHeaderTextSplitter로 `#`/`##` 섹션 단위로 먼저 나눈다 — 서문과 첫 조문이 병합되지 않도록.
2) `## 출처` 섹션은 제외한다 — URL 목록은 검색 노이즈이고 source는 메타데이터에 있다.
3) chunk_size를 넘는 섹션만 문자 기준으로 추가 분할한다.
4) 모든 청크 앞에 "[문서 제목] "을 붙인다 — 맥락이 약한 헤더 청크도 어느 문서인지 알 수 있게.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rent_agent.config import Settings
from rent_agent.rag.loader import load_markdown_docs

HEADERS = [("#", "h1"), ("##", "section")]
EXCLUDED_SECTION_PREFIXES = ("출처",)
FALLBACK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# 섹션 = 청크 원칙: 가장 긴 섹션(≈950자)이 잘리지 않도록 1000. 800이면 734자+210자로
# 갈라진 꼬리(실무 팁)가 본문보다 먼저 검색되어 정답 근거가 컨텍스트에서 빠지는 문제가
# 실측됨 (RAGAS Q9, 2026-09-02).
DEFAULT_CHUNK_SIZE = 1000


def split_documents(
    docs: list[Document], chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = 100
) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=FALLBACK_SEPARATORS
    )
    chunks: list[Document] = []
    for doc in docs:
        for sec in header_splitter.split_text(doc.page_content):
            section = str(sec.metadata.get("section", ""))
            if section.startswith(EXCLUDED_SECTION_PREFIXES):
                continue
            meta = {**doc.metadata, "section": section}
            pieces = (
                [sec]
                if len(sec.page_content) <= chunk_size
                else char_splitter.split_documents([sec])
            )
            for piece in pieces:
                chunks.append(
                    Document(
                        page_content=f"[{meta['title']}] {piece.page_content.strip()}",
                        metadata=meta,
                    )
                )
    return chunks


def chunk_ids(chunks: list[Document]) -> list[str]:
    """파일명#순번. 같은 문서를 다시 적재하면 같은 id → Chroma upsert로 중복 없음."""
    counters: dict[str, int] = {}
    ids = []
    for c in chunks:
        f = c.metadata["file"]
        counters[f] = counters.get(f, 0) + 1
        ids.append(f"{f}#{counters[f]}")
    return ids


def build_vectorstore(
    raw_dir: Path, chroma_dir: Path, embedding: Embeddings, collection: str, reset: bool = False
) -> Chroma:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    if reset:
        # 디렉터리를 지우는 대신 컬렉션만 비운다. 같은 경로를 이미 연 chromadb 클라이언트가 있으면
        # rmtree 후 쓰기에서 "attempt to write a readonly database"(code 1032)가 나기 때문.
        Chroma(
            collection_name=collection,
            embedding_function=embedding,
            persist_directory=str(chroma_dir),
        ).reset_collection()
    chunks = split_documents(load_markdown_docs(raw_dir))
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        ids=chunk_ids(chunks),
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
