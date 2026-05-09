from pathlib import Path
import sys
import argparse
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.append(str(ROOT / "src"))

from rag.config import DOCS_DIR  # noqa: E402
from rag.indexer import KnowledgeIndexer  # noqa: E402
from rag.loader import load_chunks_from_docs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG knowledge base")
    parser.add_argument(
        "--mode",
        choices=["text", "embedding", "both"],
        default="both",
        help="Knowledge base build mode",
    )
    args = parser.parse_args()

    chunks = load_chunks_from_docs(DOCS_DIR)
    indexer = KnowledgeIndexer()
    count = indexer.rebuild(chunks, mode=args.mode)
    print(f"知识库构建完成（mode={args.mode}），共入库 {count} 个文档片段。")


if __name__ == "__main__":
    main()
