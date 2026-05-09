from __future__ import annotations

import os
from typing import Sequence

import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def _resolve_ollama() -> tuple[str, str]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
    model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text").strip()
    return base_url, model


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    base_url, model = _resolve_ollama()
    endpoint_new = f"{base_url.rstrip('/')}/api/embed"
    endpoint_openai = f"{base_url.rstrip('/')}/v1/embeddings"
    endpoint_legacy = f"{base_url.rstrip('/')}/api/embeddings"

    vectors: list[list[float]] = []
    for text in texts:
        embedding = None

        # Newer Ollama API - 支持 embedding 和 embeddings 两种字段
        resp_new = requests.post(
            endpoint_new,
            json={"model": model, "input": text},
            timeout=120,
        )
        if resp_new.status_code < 400:
            data_new = resp_new.json()
            # 检查单数形式 embedding
            if "embedding" in data_new:
                embedding = data_new.get("embedding")
            # 检查复数形式 embeddings
            elif "embeddings" in data_new:
                embeddings = data_new.get("embeddings")
                if embeddings and isinstance(embeddings, list) and embeddings[0]:
                    embedding = embeddings[0]

        # OpenAI 兼容 API fallback
        if embedding is None:
            resp_openai = requests.post(
                endpoint_openai,
                json={"model": model, "input": text},
                timeout=120,
            )
            if resp_openai.status_code < 400:
                data_openai = resp_openai.json()
                data_items = data_openai.get("data")
                if data_items and isinstance(data_items, list):
                    first = data_items[0] or {}
                    embedding = first.get("embedding")

        # Legacy API fallback
        if embedding is None:
            resp_legacy = requests.post(
                endpoint_legacy,
                json={"model": model, "prompt": text},
                timeout=120,
            )
            if resp_legacy.status_code < 400:
                data_legacy = resp_legacy.json()
                embedding = data_legacy.get("embedding")

        if not embedding:
            raise ValueError(
                "Ollama embedding endpoint unavailable. Please verify your Ollama "
                "version and model, then test /api/embed or /v1/embeddings manually."
            )
        vectors.append(embedding)
    return vectors
