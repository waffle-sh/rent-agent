"""data/raw → Chroma 적재. 사용: uv run python scripts/ingest.py [--no-reset]"""

import argparse

from rent_agent.config import get_settings
from rent_agent.rag.ingest import ingest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-reset", action="store_true", help="기존 컬렉션 유지 후 추가")
    args = parser.parse_args()
    settings = get_settings()
    n = ingest(settings, reset=not args.no_reset)
    print(f"적재 완료: {n} chunks → {settings.chroma_dir}")
