from pathlib import Path

from rent_agent.rag.loader import load_markdown_docs


def test_load_markdown_with_frontmatter(tmp_path: Path):
    (tmp_path / "a.md").write_text(
        "---\ntitle: 테스트 문서\nsource: https://example.com\n"
        "effective_date: 2024-01-01\ncategory: law\n---\n"
        "## 제1조\n본문입니다.",
        encoding="utf-8",
    )
    (tmp_path / "ignore.txt").write_text("not md", encoding="utf-8")
    docs = load_markdown_docs(tmp_path)
    assert len(docs) == 1
    d = docs[0]
    assert d.page_content.startswith("## 제1조")
    assert d.metadata["title"] == "테스트 문서"
    assert d.metadata["source"] == "https://example.com"
    assert d.metadata["effective_date"] == "2024-01-01"
    assert d.metadata["category"] == "law"
    assert d.metadata["file"] == "a.md"


def test_load_missing_frontmatter_uses_filename(tmp_path: Path):
    (tmp_path / "b.md").write_text("본문만", encoding="utf-8")
    docs = load_markdown_docs(tmp_path)
    assert docs[0].metadata["title"] == "b"
    assert docs[0].metadata["source"] == ""
