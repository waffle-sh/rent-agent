from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding

from rent_agent.rag.ingest import build_vectorstore, split_documents
from rent_agent.rag.loader import load_markdown_docs


def _write_docs(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "law.md").write_text(
        "---\ntitle: 임대차법\nsource: s\ncategory: law\n---\n"
        "## 대항력\n" + "주택 인도와 전입신고를 하면 다음 날부터 대항력이 생긴다. " * 30
        + "\n## 우선변제권\n" + "확정일자를 받으면 우선변제권이 생긴다. " * 30,
        encoding="utf-8",
    )
    return raw


def test_split_respects_headers_and_keeps_metadata(tmp_path: Path):
    docs = load_markdown_docs(_write_docs(tmp_path))
    chunks = split_documents(docs, chunk_size=800, chunk_overlap=100)
    assert len(chunks) >= 2
    assert all(c.metadata["title"] == "임대차법" for c in chunks)
    # 문서 제목을 청크 앞에 붙여, 헤더만 있는 청크도 어떤 문서인지 알 수 있게 한다
    # (임베딩·LLM 모두에 유리)
    assert all(c.page_content.startswith("[임대차법] ") for c in chunks)
    assert all(len(c.page_content) <= 800 + len("[임대차법] ") for c in chunks)


def test_build_vectorstore_persists_and_searches(tmp_path: Path):
    raw = _write_docs(tmp_path)
    chroma_dir = tmp_path / "chroma"
    emb = DeterministicFakeEmbedding(size=64)
    vs = build_vectorstore(
        raw_dir=raw, chroma_dir=chroma_dir, embedding=emb, collection="test_col", reset=True
    )
    assert vs._collection.count() >= 2
    results = vs.similarity_search("대항력", k=1)
    assert len(results) == 1
    assert results[0].metadata["title"] == "임대차법"


def test_reset_clears_previous(tmp_path: Path):
    raw = _write_docs(tmp_path)
    chroma_dir = tmp_path / "chroma"
    emb = DeterministicFakeEmbedding(size=64)
    first = build_vectorstore(
        raw_dir=raw, chroma_dir=chroma_dir, embedding=emb, collection="test_col", reset=True
    )
    n = first._collection.count()
    second = build_vectorstore(
        raw_dir=raw, chroma_dir=chroma_dir, embedding=emb, collection="test_col", reset=True
    )
    assert second._collection.count() == n  # 중복 적재 없음
