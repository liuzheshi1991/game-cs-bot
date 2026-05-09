from __future__ import annotations

import json
from typing import Sequence

from .config import CHUNKS_STORE, VECTOR_STORE
from .embeddings import embed_texts
from .loader import DocumentChunk


class KnowledgeIndexer:
    def __init__(self) -> None:
        CHUNKS_STORE.parent.mkdir(parents=True, exist_ok=True)

    def rebuild(self, chunks: Sequence[DocumentChunk], mode: str = "text") -> int:
        payload = [
            {"chunk_id": c.chunk_id, "content": c.content, "source": c.source}
            for c in chunks
        ]
        if mode in {"text", "both"}:
            CHUNKS_STORE.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if mode in {"embedding", "both"}:
            texts = [item["content"] for item in payload]
            embeddings = embed_texts(texts)
            vector_payload = []
            for item, emb in zip(payload, embeddings):
                vector_payload.append(
                    {
                        "chunk_id": item["chunk_id"],
                        "content": item["content"],
                        "source": item["source"],
                        "embedding": emb,
                    }
                )
            VECTOR_STORE.write_text(
                json.dumps(vector_payload, ensure_ascii=False),
                encoding="utf-8",
            )
        return len(payload)
