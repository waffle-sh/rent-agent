from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding

from rent_agent.config import Settings
from rent_agent.rag.ingest import build_vectorstore, split_documents
from rent_agent.rag.loader import load_markdown_docs
from rent_agent.rag.retriever import get_retriever

REAL_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def _write_docs(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "law.md").write_text(
        "---\ntitle: 임대차법\nsource: s\ncategory: law\n---\n"
        "# 주택임대차보호법 핵심\n이 문서는 개요입니다.\n"
        "## 대항력\n" + "주택 인도와 전입신고를 하면 다음 날부터 대항력이 생긴다. " * 30
        + "\n## 우선변제권\n" + "확정일자를 받으면 우선변제권이 생긴다. " * 30
        + "\n## 출처\n- https://law.go.kr\n",
        encoding="utf-8",
    )
    return raw


def test_each_section_is_its_own_chunk_and_preamble_not_merged(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.md").write_text(
        "---\ntitle: 문서\n---\n# 제목\n서문 한 줄.\n## A 조문\n짧은 내용 A.\n"
        "## B 조문\n짧은 내용 B.\n## 출처\n- http://x",
        encoding="utf-8",
    )
    chunks = split_documents(load_markdown_docs(raw))
    sections = [c.metadata["section"] for c in chunks]
    # 서문(H1)과 각 ## 섹션이 병합되지 않고 각각 한 청크. 출처는 제외.
    assert sections == ["", "A 조문", "B 조문"]
    assert chunks[1].page_content == "[문서] ## A 조문\n짧은 내용 A."


def test_long_section_is_split_but_keeps_metadata(tmp_path: Path):
    docs = load_markdown_docs(_write_docs(tmp_path))
    chunks = split_documents(docs, chunk_size=800, chunk_overlap=100)
    assert len(chunks) >= 4  # 서문 1 + 대항력 ≥2 + 우선변제권 ≥2 (출처 제외)
    assert all(c.metadata["title"] == "임대차법" for c in chunks)
    assert {c.metadata["section"] for c in chunks} == {"", "대항력", "우선변제권"}
    assert all(c.page_content.startswith("[임대차법] ") for c in chunks)
    assert all(len(c.page_content) <= 800 + len("[임대차법] ") for c in chunks)
    assert not any("law.go.kr" in c.page_content for c in chunks)


def test_real_corpus_chunking_is_stable():
    # 실제 문서 5종에 대한 골든 값. 청킹 규칙이 바뀌면 의도적으로 갱신할 것.
    chunks = split_documents(load_markdown_docs(REAL_RAW))
    assert 40 <= len(chunks) <= 70
    assert not any(c.metadata["section"].startswith("출처") for c in chunks)
    assert all(len(c.page_content) <= 900 for c in chunks)


def test_build_vectorstore_persists_and_searches(tmp_path: Path):
    raw = _write_docs(tmp_path)
    chroma_dir = tmp_path / "chroma"
    emb = DeterministicFakeEmbedding(size=64)
    vs = build_vectorstore(
        raw_dir=raw, chroma_dir=chroma_dir, embedding=emb, collection="test_col", reset=True
    )
    n = vs._collection.count()
    assert n >= 4
    # 디스크에서 다시 열어도 같은 개수 → 실제로 persist 됨
    reopened = Chroma(
        collection_name="test_col", embedding_function=emb, persist_directory=str(chroma_dir)
    )
    assert reopened._collection.count() == n
    results = reopened.similarity_search("대항력", k=1)
    assert results[0].metadata["title"] == "임대차법"


def test_reingest_is_idempotent_with_and_without_reset(tmp_path: Path):
    raw = _write_docs(tmp_path)
    chroma_dir = tmp_path / "chroma"
    emb = DeterministicFakeEmbedding(size=64)
    n = build_vectorstore(raw, chroma_dir, emb, "test_col", reset=True)._collection.count()
    assert build_vectorstore(raw, chroma_dir, emb, "test_col", reset=True)._collection.count() == n
    # 결정적 id(file#i) 업서트 → reset=False로 다시 적재해도 중복 없음
    assert build_vectorstore(raw, chroma_dir, emb, "test_col", reset=False)._collection.count() == n


def test_retriever_uses_plain_similarity_with_k(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RETRIEVER_K", "3")
    retriever = get_retriever(Settings(), embedding=DeterministicFakeEmbedding(size=8))
    assert retriever.search_type == "similarity"
    assert retriever.search_kwargs == {"k": 3}
