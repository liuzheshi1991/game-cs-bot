from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from rag.qa import ask_with_rag  # noqa: E402


def run_once(question: str, top_k: int, mode: str) -> dict:
    return ask_with_rag(question, top_k=top_k, retrieval_mode=mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare text vs embedding retrieval")
    parser.add_argument("question", help="User question for comparison")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks per mode")
    args = parser.parse_args()

    text_result = run_once(args.question, args.top_k, "text")
    try:
        embedding_result = run_once(args.question, args.top_k, "embedding")
    except Exception as exc:
        embedding_result = {
            "answer": f"embedding 模式执行失败：{exc}",
            "context": "embedding 索引或接口不可用，请先执行 build_kb --mode embedding 并确认 Ollama embedding 接口可访问。",
        }

    print("\n==============================")
    print("问题:", args.question)
    print("==============================")

    print("\n[TEXT 检索] 回答")
    print(text_result["answer"])
    print("\n[TEXT 检索] 上下文")
    print(text_result["context"])

    print("\n------------------------------")

    print("\n[EMBEDDING 检索] 回答")
    print(embedding_result["answer"])
    print("\n[EMBEDDING 检索] 上下文")
    print(embedding_result["context"])


if __name__ == "__main__":
    main()
