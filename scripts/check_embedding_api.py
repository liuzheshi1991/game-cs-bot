from __future__ import annotations

import argparse
import json
from typing import Any

import requests


def _print_json(title: str, data: Any) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _check_tags(base_url: str) -> tuple[bool, list[str]]:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = [item.get("name", "") for item in data.get("models", [])]
        _print_json("Ollama 模型列表", models)
        return True, models
    except Exception as exc:
        print(f"\n[FAIL] 无法访问 {url}: {exc}")
        return False, []


def _try_endpoint(name: str, url: str, payload: dict[str, Any]) -> tuple[bool, str]:
    try:
        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code >= 400:
            return False, f"HTTP {resp.status_code}"
        data = resp.json()
        return True, json.dumps(data, ensure_ascii=False)[:300]
    except Exception as exc:
        return False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Ollama embedding endpoints")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Ollama base URL",
    )
    parser.add_argument(
        "--model",
        default="nomic-embed-text",
        help="Embedding model name to test",
    )
    parser.add_argument(
        "--text",
        default="这是一个 embedding 连通性测试",
        help="Test input text",
    )
    args = parser.parse_args()

    print(f"[INFO] base_url={args.base_url}")
    print(f"[INFO] model={args.model}")

    ok, models = _check_tags(args.base_url)
    if not ok:
        print("\n建议：先启动 Ollama 服务，例如 `ollama serve`。")
        return

    if args.model not in models:
        print(f"\n[WARN] 模型 `{args.model}` 不在当前模型列表中。")
        print(f"建议先拉取：`ollama pull {args.model}`")

    endpoints = [
        (
            "api/embed",
            f"{args.base_url.rstrip('/')}/api/embed",
            {"model": args.model, "input": args.text},
        ),
        (
            "v1/embeddings",
            f"{args.base_url.rstrip('/')}/v1/embeddings",
            {"model": args.model, "input": args.text},
        ),
        (
            "api/embeddings",
            f"{args.base_url.rstrip('/')}/api/embeddings",
            {"model": args.model, "prompt": args.text},
        ),
    ]

    any_success = False
    for name, url, payload in endpoints:
        success, detail = _try_endpoint(name, url, payload)
        if success:
            any_success = True
            print(f"\n[OK] {name} 可用: {url}")
            print(f"响应摘要: {detail}")
        else:
            print(f"\n[FAIL] {name} 不可用: {url}")
            print(f"原因: {detail}")

    if any_success:
        print("\n结论：检测到至少一个 embedding 接口可用，可以启用 embedding 检索模式。")
    else:
        print("\n结论：未检测到可用 embedding 接口。")
        print("建议：升级 Ollama 到较新版本后重试，或继续使用 text 检索模式。")


if __name__ == "__main__":
    main()
