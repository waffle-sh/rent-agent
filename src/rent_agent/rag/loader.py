"""data/raw/*.md → langchain Document. frontmatter를 metadata로 옮긴다."""

from pathlib import Path

import frontmatter
from langchain_core.documents import Document


def load_markdown_docs(raw_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(raw_dir.glob("*.md")):
        post = frontmatter.load(path, encoding="utf-8")
        # Chroma 메타데이터는 str/int/float/bool만 허용 → 전부 str.
        # 키가 있지만 값이 null이면 "None"이 아니라 ""로.
        meta = {
            "title": str(post.get("title") or path.stem),
            "source": str(post.get("source") or ""),
            "effective_date": str(post.get("effective_date") or ""),
            "category": str(post.get("category") or ""),
            "file": path.name,
        }
        docs.append(Document(page_content=post.content.strip(), metadata=meta))
    return docs
