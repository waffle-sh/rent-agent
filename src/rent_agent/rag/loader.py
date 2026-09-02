"""data/raw/*.md → langchain Document. frontmatter를 metadata로 옮긴다."""

from pathlib import Path

import frontmatter
from langchain_core.documents import Document


def load_markdown_docs(raw_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(raw_dir.glob("*.md")):
        post = frontmatter.load(path, encoding="utf-8")
        # Chroma metadata는 str/int/float/bool만 받는다. YAML이 date로 파싱하는
        # effective_date 등을 포함해 모두 str로 정규화한다.
        meta = {
            "title": str(post.get("title", path.stem)),
            "source": str(post.get("source", "")),
            "effective_date": str(post.get("effective_date", "")),
            "category": str(post.get("category", "")),
            "file": path.name,
        }
        docs.append(Document(page_content=post.content.strip(), metadata=meta))
    return docs
