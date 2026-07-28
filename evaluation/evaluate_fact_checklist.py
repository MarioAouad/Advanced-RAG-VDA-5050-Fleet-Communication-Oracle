
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "test_dataset.json"
RESULTS_DIR = SCRIPT_DIR / "results"

# 1. Load dataset
def load_dataset() -> dict:

    if not DATASET_PATH.exists():
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    test_data = dataset.get("test_data", [])
    print(f"[OK] Loaded {len(test_data)} questions from {DATASET_PATH.name}")
    return dataset

# 2. Scoring functions
def score_fact_accuracy(answer: str, expected_keywords: list[str]) -> float:

    if not expected_keywords:
        return 1.0  # No keywords to check — full score

    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)

def score_retrieval_hit_rate(sources: list[str], expected_sources: list[str]) -> float:

    if not expected_sources:
        return 1.0  # No sources to check — full score

    # Flatten all source paths into a single searchable string
    all_sources = " ".join(sources).lower()
    hits = sum(1 for src in expected_sources if src.lower() in all_sources)
    return hits / len(expected_sources)

# 3. Run evaluation
def run_evaluation(test_data: list[dict], limit: int | None = None) -> list[dict]:

    from backend.core.generator import query_rag

    if limit:
        test_data = test_data[:limit]

    results = []
    total = len(test_data)

    for i, entry in enumerate(test_data, 1):
        qid = entry["id"]
        question = entry["question"]
        expected_keywords = entry.get("expected_keywords", [])
        expected_sources = entry.get("expected_sources", [])

        print(f"  [{i}/{total}] {qid}: {question[:60]}...")

        try:
            rag_result = query_rag(question)
            answer = rag_result["answer"]
            contexts = rag_result["contexts"]
            sources = rag_result.get("sources", [])

            fact_score = score_fact_accuracy(answer, expected_keywords)
            retrieval_score = score_retrieval_hit_rate(sources, expected_sources)

            # Identify which keywords were missed
            answer_lower = answer.lower()
            missed_keywords = [kw for kw in expected_keywords if kw.lower() not in answer_lower]

            results.append({
                "id": qid,
                "category": entry.get("category", "unknown"),
                "fact_accuracy": round(fact_score * 100, 1),
                "retrieval_hit_rate": round(retrieval_score * 100, 1),
                "missed_keywords": missed_keywords,
                "keywords_total": len(expected_keywords),
                "keywords_found": len(expected_keywords) - len(missed_keywords),
            })

            status = "✓" if fact_score == 1.0 else "~"
            print(f"         {status} Fact Accuracy: {fact_score*100:.0f}%  |  Retrieval Hit: {retrieval_score*100:.0f}%")

        except Exception as e:
            print(f"         ✗ ERROR: {type(e).__name__}: {e}")
            results.append({
                "id": qid,
                "category": entry.get("category", "unknown"),
                "fact_accuracy": 0.0,
                "retrieval_hit_rate": 0.0,
                "missed_keywords": expected_keywords,
                "keywords_total": len(expected_keywords),
                "keywords_found": 0,
                "error": str(e),
            })

    return results

# 4. Display & persist results
def print_summary(results: list[dict]) -> None:

    if not results:
        print("\n[WARN] No results to display.")
        return

    avg_fact = sum(r["fact_accuracy"] for r in results) / len(results)
    avg_retrieval = sum(r["retrieval_hit_rate"] for r in results) / len(results)

    # Per-category breakdown
    categories: dict[str, list[dict]] = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, []).append(r)

    print("\n" + "=" * 70)
    print("  Fact-Checklist Evaluation Results")
    print("=" * 70)

    # Overall metrics
    print(f"\n  {'Metric':<30} {'Score':>10}")
    print(f"  {'-'*30} {'-'*10}")
    print(f"  {'Avg Fact Accuracy (%)':<30} {avg_fact:>9.1f}%")
    print(f"  {'Avg Retrieval Hit Rate (%)':<30} {avg_retrieval:>9.1f}%")
    print(f"  {'Questions Evaluated':<30} {len(results):>10}")

    # Per-category breakdown
    print(f"\n  {'Category':<40} {'Fact Acc':>10} {'Retr Hit':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10}")
    for cat, cat_results in sorted(categories.items()):
        cat_fact = sum(r["fact_accuracy"] for r in cat_results) / len(cat_results)
        cat_retr = sum(r["retrieval_hit_rate"] for r in cat_results) / len(cat_results)
        print(f"  {cat:<40} {cat_fact:>9.1f}% {cat_retr:>9.1f}%")

    # Per-question detail
    print(f"\n  {'ID':<8} {'Fact Acc':>10} {'Retr Hit':>10} {'Missed Keywords'}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*30}")
    for r in results:
        missed = ", ".join(r["missed_keywords"][:3]) if r["missed_keywords"] else "—"
        if len(r["missed_keywords"]) > 3:
            missed += f" (+{len(r['missed_keywords'])-3} more)"
        print(f"  {r['id']:<8} {r['fact_accuracy']:>9.1f}% {r['retrieval_hit_rate']:>9.1f}%  {missed}")

    print("\n" + "=" * 70 + "\n")

def save_results(results: list[dict], tag: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"fact_checklist_{tag}.json"

    avg_fact = sum(r["fact_accuracy"] for r in results) / len(results) if results else 0
    avg_retrieval = sum(r["retrieval_hit_rate"] for r in results) / len(results) if results else 0

    payload = {
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_method": "fact_checklist",
        "summary": {
            "avg_fact_accuracy": round(avg_fact, 1),
            "avg_retrieval_hit_rate": round(avg_retrieval, 1),
            "questions_evaluated": len(results),
        },
        "per_question": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] Results saved to: {output_path}")

# 5. CLI entry point
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic Fact-Checklist evaluation for the VDA-5050 RAG pipeline."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="baseline",
        help="Label for this evaluation run (default: 'baseline').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions to evaluate (for quick testing).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  VDA-5050 Fleet Communication Oracle — Fact-Checklist Evaluation")
    print("=" * 70)

    dataset = load_dataset()
    test_data = dataset["test_data"]

    print(f"\n[EVAL] Evaluating {args.limit or len(test_data)} questions (tag: '{args.tag}')...\n")

    results = run_evaluation(test_data, limit=args.limit)

    print_summary(results)
    save_results(results, args.tag)

    print(f"[OK] Evaluation complete. Run tag: '{args.tag}'")

if __name__ == "__main__":
    main()
