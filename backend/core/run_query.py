from __future__ import annotations
import logging
import sys
import textwrap
from pathlib import Path

# Ensure project root is on sys.path when running as a standalone script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.core.generator import query_rag

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("query")

# Default test question (from our evaluation dataset Category A)
_DEFAULT_QUESTION = (
    "What is the recommended MQTT topic structure for a local broker "
    "in VDA 5050, and can you provide a concrete example?"
)

_EXIT_COMMANDS = {"exit", "quit", "q"}

# Display helpers
def _print_result(result: dict) -> None:

    print("\n" + "=" * 70)
    print("  QUESTION")
    print("=" * 70)
    print(textwrap.fill(result["question"], width=68, initial_indent="  ", subsequent_indent="  "))

    print("\n" + "=" * 70)
    print("  ANSWER")
    print("=" * 70)
    # Wrap long answers for readability
    for line in result["answer"].split("\n"):
        print(textwrap.fill(line, width=68, initial_indent="  ", subsequent_indent="  "))

    print("\n" + "=" * 70)
    print(f"  RETRIEVED CONTEXTS ({len(result['contexts'])} chunks)")
    print("=" * 70)
    for i, ctx in enumerate(result["contexts"], 1):
        preview = textwrap.shorten(ctx, width=200, placeholder=" [...]")
        print(f"\n  [{i}] ({len(ctx)} chars)")
        print(f"  {preview}")

    print("\n" + "=" * 70 + "\n")

# Main — Interactive CLI Loop
def main() -> None:
    print("\n" + "=" * 70)
    print("  VDA-5050 RAG — Interactive Query Tool")
    print("=" * 70)

    # Single-shot mode: question passed as CLI argument
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\n  Querying RAG pipeline...")
        print(f"  Question: {question[:80]}{'...' if len(question) > 80 else ''}")
        try:
            result = query_rag(question)
            _print_result(result)
        except RuntimeError as e:
            print(f"\n  [ERROR] {e}")
            print("  Make sure you have:")
            print("    1. Run ingestion first: python backend/core/run_ingestion.py")
            print("    2. Set a valid GROQ_API_KEY in your .env file")
            sys.exit(1)
        except Exception as e:
            logger.exception("Unexpected error during query.")
            print(f"\n  [ERROR] {type(e).__name__}: {e}")
            sys.exit(1)
        return

    # Interactive loop mode
    print(f"\n  Type your question and press Enter.")
    print(f"  Type 'exit', 'quit', or 'q' to leave.\n")

    while True:
        try:
            user_input = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in _EXIT_COMMANDS:
            print("\n  Goodbye!")
            break

        print(f"\n  Querying RAG pipeline...")
        print(f"  Question: {user_input[:80]}{'...' if len(user_input) > 80 else ''}")

        try:
            result = query_rag(user_input)
            _print_result(result)
        except RuntimeError as e:
            print(f"\n  [ERROR] {e}")
            print("  Make sure you have:")
            print("    1. Run ingestion first: python backend/core/run_ingestion.py")
            print("    2. Set a valid GROQ_API_KEY in your .env file\n")
        except Exception as e:
            logger.exception("Unexpected error during query.")
            print(f"\n  [ERROR] {type(e).__name__}: {e}\n")

if __name__ == "__main__":
    main()
