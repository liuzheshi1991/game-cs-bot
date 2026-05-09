from pathlib import Path
import sys
import argparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from rag.qa import ask_with_rag  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask with RAG")
    parser.add_argument("question", help="User question")
    parser.add_argument(
        "--mode",
        choices=["text", "embedding"],
        default="text",
        help="Retrieval mode",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks")
    args = parser.parse_args()

    import json
    
    result = ask_with_rag(args.question, top_k=args.top_k, retrieval_mode=args.mode)
    print(f"\n=== 检索模式: {result['retrieval_mode']} ===")
    print("\n=== 机器人回答 ===")
    print(result["answer"])
    print("\n=== 检索到的上下文 ===")
    print(result["context"])
    print("\n=== 用户 Prompt ===")
    print(result["prompt"])
    
    # 输出实际发送给 Ollama API 的请求内容
    if result.get("api_payload"):
        print("\n=== Ollama API 请求内容 ===")
        print(json.dumps(result["api_payload"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
