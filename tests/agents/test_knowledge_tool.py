from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding

from rent_agent.agents.knowledge_agent import make_knowledge_tool
from rent_agent.rag.ingest import build_vectorstore


def test_search_tool_formats_sources(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.md").write_text(
        "---\ntitle: 임대차법\nsource: https://law.go.kr\neffective_date: 2024-01-01\n---\n"
        "## 대항력\n전입신고 다음 날 효력.",
        encoding="utf-8",
    )
    vs = build_vectorstore(
        raw, tmp_path / "chroma", DeterministicFakeEmbedding(size=32), "test_docs", reset=True
    )
    search = make_knowledge_tool(vs.as_retriever(search_kwargs={"k": 1}))
    out = search.invoke({"query": "대항력"})
    assert "[출처: 임대차법 | https://law.go.kr | 기준일 2024-01-01]" in out
    assert "전입신고 다음 날 효력" in out
