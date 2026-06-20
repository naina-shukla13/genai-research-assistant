"""
Runs the evaluation dataset against the live system and prints a
pass/fail table, including whether RAG was correctly used or avoided.

This produces the "small evaluation dataset (5–10 queries)" bonus
deliverable as a runnable artifact, not just a static JSON file nobody
executes.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.coordinator import coordinator
from utils.logger import log_event


def load_eval_set(path: str = "evaluation/eval_dataset.json") -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation() -> None:
    eval_cases = load_eval_set()
    results = []

    print(f"Running {len(eval_cases)} evaluation cases...\n")

    for case in eval_cases:
        response = coordinator.handle_query(case["query"])

        intent_match = response.intent.value == case["expected_intent"]
        rag_used = len(response.retrieved_chunks) > 0
        rag_match = rag_used == case["should_use_rag"]
        passed = intent_match and rag_match

        results.append({
            "id": case["id"],
            "query": case["query"],
            "expected_intent": case["expected_intent"],
            "actual_intent": response.intent.value,
            "expected_rag": case["should_use_rag"],
            "actual_rag": rag_used,
            "passed": passed,
        })

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {case['id']} | intent: {response.intent.value:<20} | rag_used: {rag_used}")
        if not passed:
            print(f"        expected_intent={case['expected_intent']}, expected_rag={case['should_use_rag']}")
        print(f"        query: {case['query']}")
        print()

    passed_count = sum(r["passed"] for r in results)
    print("=" * 70)
    print(f"RESULT: {passed_count}/{len(results)} passed")
    print("=" * 70)

    log_event("Evaluation", "eval_run_completed", passed=passed_count, total=len(results))

    with open("evaluation/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_evaluation()