from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "data" / "chroma"
COLLECTION_NAME = "game_cs_kb"
CHUNKS_STORE = BASE_DIR / "data" / "kb_chunks.json"
VECTOR_STORE = BASE_DIR / "data" / "kb_vectors.json"

# 经验值：客服 FAQ 文档通常是短句，chunk 不宜过大，但需要足够大以容纳完整的问答
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
