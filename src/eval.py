"""
Step 3: a basic eval harness.

A fixed set of test questions with expected answers, run through the full
pipeline (ask.py), and scored automatically. Half the questions need the
retrieved write-up (RAG), half need a real SQL query (tool use) — on purpose,
to show both paths get exercised and checked.

Grading: for numeric answers, pass if a number within `tolerance` of the
expected value appears anywhere in the response. For text answers, pass if
all of the required keywords appear (case-insensitive).

Usage:
    python src/eval.py
    python src/eval.py --verbose
"""
import argparse
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from ask import ask  # noqa: E402

TEST_CASES = [
    # --- Questions answerable from the retrieved write-up (RAG) ---
    {
        "question": "Does sentiment improve as temperature increases from winter to spring?",
        "type": "keywords",
        "expect": ["yes", "increase"],
        "source": "rag",
    },
    {
        "question": "What were the three main findings summarized in the report?",
        "type": "keywords",
        "expect": ["warmer", "negative", "light"],
        "source": "rag",
    },
    {
        "question": "What are the suggested directions for further research on this project?",
        "type": "keywords",
        "expect": ["lag", "dataset", "outlier"],
        "source": "rag",
    },
    {
        "question": "What data sources were used to build this project?",
        "type": "keywords",
        "expect": ["reddit", "open-meteo"],
        "source": "rag",
    },
    # --- Questions that need a real SQL query (tool use) ---
    {
        "question": "How many total Reddit posts/comments were scored in this dataset?",
        "type": "numeric",
        "expect": 87725,
        "tolerance": 5,
        "source": "tool",
    },
    {
        "question": "What was the average temperature on days with positive average sentiment?",
        "type": "numeric",
        "expect": 9.96,
        "tolerance": 0.5,
        "source": "tool",
    },
    {
        "question": "What was the average temperature on days with negative average sentiment?",
        "type": "numeric",
        "expect": 2.99,
        "tolerance": 0.5,
        "source": "tool",
    },
    {
        "question": "What is the average sentiment score on low light-intensity days?",
        "type": "numeric",
        "expect": -0.022,
        "tolerance": 0.01,
        "source": "tool",
    },
    {
        "question": "What is the average sentiment score on high light-intensity days?",
        "type": "numeric",
        "expect": -0.0014,
        "tolerance": 0.01,
        "source": "tool",
    },
    {
        "question": "How many days of data are in the daily sentiment/weather table?",
        "type": "numeric",
        "expect": 129,
        "tolerance": 1,
        "source": "tool",
    },
]


def extract_numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]


def grade(case: dict, answer: str) -> bool:
    if case["type"] == "numeric":
        nums = extract_numbers(answer)
        return any(abs(n - case["expect"]) <= case["tolerance"] for n in nums)
    elif case["type"] == "keywords":
        lower = answer.lower()
        return all(kw.lower() in lower for kw in case["expect"])
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = []
    for i, case in enumerate(TEST_CASES, 1):
        result = ask(case["question"], verbose=False)
        passed = grade(case, result["answer"])
        results.append(passed)

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] ({case['source']}) {case['question']}")
        if args.verbose or not passed:
            print(f"    expected: {case.get('expect')}")
            print(f"    got: {result['answer'][:300]}")
        print()

    n_pass = sum(results)
    n_total = len(results)
    rag_results = [r for r, c in zip(results, TEST_CASES) if c["source"] == "rag"]
    tool_results = [r for r, c in zip(results, TEST_CASES) if c["source"] == "tool"]

    print("=" * 50)
    print(f"Overall:    {n_pass}/{n_total} passed ({100*n_pass/n_total:.0f}%)")
    print(f"RAG-only:   {sum(rag_results)}/{len(rag_results)} passed")
    print(f"Tool-use:   {sum(tool_results)}/{len(tool_results)} passed")


if __name__ == "__main__":
    main()
