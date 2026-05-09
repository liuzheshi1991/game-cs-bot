from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass
class DocumentChunk:
    chunk_id: str
    content: str
    source: str


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(0, end - overlap)

    return chunks


def _load_markdown_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_chunks_from_docs(docs_dir: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    md_files = sorted(docs_dir.glob("*.md"))

    for file_path in md_files:
        text = _load_markdown_file(file_path)
        split_chunks = _chunk_text(text)
        for idx, chunk in enumerate(split_chunks):
            chunk_id = f"{file_path.stem}-{idx}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk,
                    source=file_path.name,
                )
            )
    return chunks


def format_context(chunks: Iterable[DocumentChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[片段{i}] 来源: {chunk.source}\n{chunk.content}")
    return "\n\n".join(parts)
