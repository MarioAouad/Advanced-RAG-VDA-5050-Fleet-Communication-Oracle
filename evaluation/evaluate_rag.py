"""
VDA-5050 Fleet Communication Oracle — RAG Evaluation Pipeline
=============================================================
This script loads the ground-truth evaluation dataset and runs the RAGAS
evaluation framework to measure retrieval and generation quality.

Metrics computed:
    • Context Precision  — Are the retrieved chunks ranked with relevant ones first?
    • Context Recall     — Does the retrieved context cover the ground-truth answer?
    • Faithfulness       — Is the generated answer grounded in the retrieved context?
    • Answer Relevancy   — Is the generated answer relevant to the original question?

Usage:
    # Baseline run (before improvements):
    python evaluation/evaluate_rag.py --tag baseline

    # After applying improvements:
    python evaluation/evaluate_rag.py --tag improved_v1

    # Compare two saved runs:
    python evaluation/evaluate_rag.py --compare baseline improved_v1

The script persists each run's results to evaluation/results/<tag>.json so that
you can always go back and compare baseline vs. improved metrics for the report.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional: suppress noisy transformer warnings during import
# ---------------------------------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_PATH = SCRIPT_DIR / "test_dataset.json"
RESULTS_DIR = SCRIPT_DIR / "results"

# ---------------------------------------------------------------------------
# Helper: pretty table printer (no external dependency)
# ---------------------------------------------------------------------------

def _print_table(headers: list[str], rows: list[list[str]], title: str = "") -> None:
    """Print a simple ASCII table to stdout."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    if title:
        print(f"\n{'=' * len(sep)}")
        print(f"  {title}")
        print(f"{'=' * len(sep)}")
    print(sep)
    print(header_row)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |")
    print(sep)


# ---------------------------------------------------------------------------
# 1. Load & validate the ground-truth dataset
# ---------------------------------------------------------------------------

def load_dataset(path: Path = DATASET_PATH) -> dict:
    """Load the ground-truth test dataset and perform basic validation."""
    if not path.exists():
        print(f"[ERROR] Dataset not found at: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    test_data = dataset.get("test_data", [])
    if not test_data:
        print("[ERROR] 'test_data' array is empty or missing in the dataset.")
        sys.exit(1)

    # Validate required fields per entry
    required_keys = {"id", "category", "question", "ground_truth_answer", "ground_truth_contexts"}
    for entry in test_data:
        missing = required_keys - set(entry.keys())
        if missing:
            print(f"[WARNING] Entry '{entry.get('id', '?')}' is missing keys: {missing}")

    # Print summary
    categories = {}
    for entry in test_data:
        cat = entry.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n[OK] Loaded {len(test_data)} evaluation questions from: {path.name}")
    _print_table(
        ["Category", "Count"],
        [[cat, str(count)] for cat, count in sorted(categories.items())],
        title="Dataset Distribution",
    )
    return dataset


# ---------------------------------------------------------------------------
# 2. Query the RAG pipeline (stub — to be connected to the FastAPI backend)
# ---------------------------------------------------------------------------

def query_rag_pipeline(question: str, backend_url: str = "http://127.0.0.1:8000") -> dict:
    """
    Send a question to the RAG backend and return the structured response.

    Returns:
        {
            "answer": str,          # The generated answer
            "contexts": list[str],  # Retrieved context chunks
        }

    NOTE: This is a *stub*. During baseline evaluation (before the backend is
    built), it returns empty placeholders so that the evaluation framework
    structure can be verified. Replace the body of this function once the
    FastAPI /query endpoint is operational.
    """
    # ------------------------------------------------------------------
    # PHASE 1 (current): Return empty stubs for framework validation
    # ------------------------------------------------------------------
    # TODO: Uncomment the block below once the backend is running.
    #
    # import requests
    # try:
    #     resp = requests.post(
    #         f"{backend_url}/query",
    #         json={"question": question},
    #         timeout=60,
    #     )
    #     resp.raise_for_status()
    #     data = resp.json()
    #     return {
    #         "answer": data.get("answer", ""),
    #         "contexts": data.get("contexts", []),
    #     }
    # except requests.RequestException as e:
    #     print(f"  [WARN] Backend call failed: {e}")
    #     return {"answer": "", "contexts": []}
    # ------------------------------------------------------------------

    return {
        "answer": "[PLACEHOLDER — backend not yet connected]",
        "contexts": ["[PLACEHOLDER — no retrieval performed yet]"],
    }


# ---------------------------------------------------------------------------
# 3. Build the RAGAS evaluation dataset
# ---------------------------------------------------------------------------

def build_ragas_dataset(test_data: list[dict], backend_url: str) -> dict:
    """
    Iterate over the ground-truth entries, query the RAG pipeline for each,
    and assemble the data structure required by RAGAS.

    Returns a dict with lists: questions, answers, contexts, ground_truths.
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    metadata_list = []

    total = len(test_data)
    for idx, entry in enumerate(test_data, 1):
        q = entry["question"]
        print(f"  [{idx}/{total}] Querying: {q[:80]}...")

        result = query_rag_pipeline(q, backend_url=backend_url)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(entry["ground_truth_answer"])
        metadata_list.append({
            "id": entry["id"],
            "category": entry["category"],
            "difficulty": entry.get("difficulty", "unknown"),
            "expected_diagram": entry.get("expected_diagram"),
        })

    return {
        "questions": questions,
        "answers": answers,
        "contexts": contexts,
        "ground_truths": ground_truths,
        "metadata": metadata_list,
    }


# ---------------------------------------------------------------------------
# 4. Run RAGAS evaluation
# ---------------------------------------------------------------------------

def run_ragas_evaluation(ragas_data: dict) -> dict:
    """
    Execute the RAGAS evaluation using the four core metrics.

    Returns a dict of metric_name → float score.

    If RAGAS / required LLM dependencies are not installed or configured, this
    function falls back to returning NaN placeholders and prints setup
    instructions.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        print(f"\n[WARN] Could not import RAGAS / datasets: {e}")
        print("   Install with: pip install ragas datasets")
        print("   Returning NaN placeholders.\n")
        return {
            "context_precision": float("nan"),
            "context_recall": float("nan"),
            "faithfulness": float("nan"),
            "answer_relevancy": float("nan"),
        }

    # Build a HuggingFace Dataset
    hf_dataset = Dataset.from_dict({
        "question": ragas_data["questions"],
        "answer": ragas_data["answers"],
        "contexts": ragas_data["contexts"],
        "ground_truth": ragas_data["ground_truths"],
    })

    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]

    # -------------------------------------------------------------------
    # RAGAS requires an LLM and embedding model for evaluation.
    # By default it uses OpenAI. For our project we route through
    # OpenRouter (OpenAI-compatible endpoint) pointed at Gemini Flash.
    #
    # Configure the LLM wrapper here once the backend is built.
    # For now, we let RAGAS use its default (requires OPENAI_API_KEY).
    #
    # Example with LangChain + OpenRouter:
    #   from langchain_openai import ChatOpenAI
    #   llm = ChatOpenAI(
    #       model="google/gemini-2.5-flash",
    #       openai_api_base="https://openrouter.ai/api/v1",
    #       openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    #   )
    #   result = evaluate(hf_dataset, metrics=metrics, llm=llm)
    # -------------------------------------------------------------------

    try:
        result = evaluate(hf_dataset, metrics=metrics)
        return dict(result)
    except Exception as e:
        print(f"\n[WARN] RAGAS evaluation failed: {e}")
        print("   This is expected if no LLM API key is configured yet.")
        print("   Returning NaN placeholders.\n")
        return {
            "context_precision": float("nan"),
            "context_recall": float("nan"),
            "faithfulness": float("nan"),
            "answer_relevancy": float("nan"),
        }


# ---------------------------------------------------------------------------
# 5. Persist results
# ---------------------------------------------------------------------------

def save_results(tag: str, scores: dict, ragas_data: dict) -> Path:
    """Save evaluation results to a JSON file under evaluation/results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{tag}.json"

    payload = {
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scores": scores,
        "per_question": [],
    }
    for i, meta in enumerate(ragas_data["metadata"]):
        payload["per_question"].append({
            **meta,
            "question": ragas_data["questions"][i],
            "generated_answer": ragas_data["answers"][i],
            "retrieved_contexts": ragas_data["contexts"][i],
            "ground_truth": ragas_data["ground_truths"][i],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[SAVED] Results saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 6. Compare two runs
# ---------------------------------------------------------------------------

def compare_runs(tag_a: str, tag_b: str) -> None:
    """Load two saved result files and print a side-by-side comparison."""
    path_a = RESULTS_DIR / f"{tag_a}.json"
    path_b = RESULTS_DIR / f"{tag_b}.json"

    for p in (path_a, path_b):
        if not p.exists():
            print(f"[ERROR] Results file not found: {p}")
            sys.exit(1)

    with open(path_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(path_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    scores_a = data_a["scores"]
    scores_b = data_b["scores"]
    all_metrics = sorted(set(scores_a) | set(scores_b))

    rows = []
    for metric in all_metrics:
        val_a = scores_a.get(metric, float("nan"))
        val_b = scores_b.get(metric, float("nan"))
        try:
            delta = val_b - val_a
            delta_str = f"{delta:+.4f}"
        except TypeError:
            delta_str = "N/A"
        rows.append([metric, f"{val_a:.4f}", f"{val_b:.4f}", delta_str])

    _print_table(
        ["Metric", f"{tag_a}", f"{tag_b}", "Delta"],
        rows,
        title=f"Comparison: {tag_a} → {tag_b}",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VDA-5050 Oracle — RAG Evaluation Pipeline (RAGAS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/evaluate_rag.py --tag baseline
  python evaluation/evaluate_rag.py --tag improved_v1 --backend http://127.0.0.1:8000
  python evaluation/evaluate_rag.py --compare baseline improved_v1
  python evaluation/evaluate_rag.py --dry-run
        """,
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="baseline",
        help="A short label for this evaluation run (e.g., 'baseline', 'improved_v1').",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="http://127.0.0.1:8000",
        help="URL of the FastAPI backend (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("TAG_A", "TAG_B"),
        help="Compare two previously saved evaluation runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the dataset and validate its structure without querying the backend.",
    )
    args = parser.parse_args()

    # --- Compare mode ---
    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
        return

    # --- Load & validate ---
    print("\n" + "=" * 70)
    print("  VDA-5050 Fleet Communication Oracle — Evaluation Pipeline")
    print("=" * 70)

    dataset = load_dataset()
    test_data = dataset["test_data"]

    if args.dry_run:
        print("\n[DONE] Dry run complete. Dataset structure is valid.")
        return

    # --- Query & Evaluate ---
    print(f"\n[QUERY] Querying RAG pipeline ({args.backend}) for {len(test_data)} questions...\n")
    ragas_data = build_ragas_dataset(test_data, backend_url=args.backend)

    print("\n[EVAL] Running RAGAS evaluation...\n")
    scores = run_ragas_evaluation(ragas_data)

    # --- Display results ---
    _print_table(
        ["Metric", "Score"],
        [[m, f"{s:.4f}"] for m, s in sorted(scores.items())],
        title=f"Evaluation Results — [{args.tag}]",
    )

    # --- Save ---
    save_results(args.tag, scores, ragas_data)

    print("\n[OK] Evaluation complete.")
    print(f"   Run tag: '{args.tag}'")
    print(f"   To compare later: python evaluation/evaluate_rag.py --compare {args.tag} <other_tag>\n")


if __name__ == "__main__":
    main()
