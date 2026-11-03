"""
Main entry point: ask a question, get an answer grounded either in the
retrieved project write-up (RAG) or in a real SQL query against the actual
data (tool use) — the model decides which it needs, and can use both.

Usage:
    python src/ask.py "What is the average temperature on positive-sentiment days?"
    python src/ask.py "What did the report say about further research?"
    python src/ask.py --verbose "Compare average sentiment across light intensity levels"

Requires: pip install anthropic openai numpy
Requires env vars: ANTHROPIC_API_KEY, OPENAI_API_KEY (embeddings only)
"""
import argparse
import json
import os
import sys

from anthropic import Anthropic
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from retrieve import Retriever  # noqa: E402
from tools import TOOL_SCHEMA, query_sentiment_weather  # noqa: E402

CLAUDE_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You are an assistant answering questions about a specific \
data analysis project: "Temperature and Light Effects on Mood Trends" \
(Reddit sentiment in r/nyc vs. NYC weather, Jan 1 - May 9 2025), studied as \
a lens on Seasonal Affective Disorder (SAD).

You have two sources of truth, and must not answer from general knowledge:

1. RETRIEVED_CONTEXT below — excerpts from the project's own write-up \
(methodology, findings, pre-computed answers). Use this for questions about \
what the report says, concludes, or recommends.

2. The `query_sentiment_weather` tool — runs real SQL against the actual \
merged dataset (daily sentiment averages + weather, and the raw 87,725 \
scored posts). Use this whenever a question needs a number, aggregate, \
comparison, or filter that isn't already given verbatim in the retrieved \
context, or to verify a number from the context against the real data.

If neither source answers the question, say so plainly rather than guessing. \
Always mention which source(s) you used (retrieved write-up vs. live query)."""


def run_tool(tool_name: str, tool_input: dict) -> dict:
    if tool_name == "query_sentiment_weather":
        return query_sentiment_weather(tool_input["sql"])
    return {"error": f"unknown tool {tool_name}"}


def ask(question: str, verbose: bool = False) -> dict:
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    retriever = Retriever(openai_client)
    hits = retriever.top_k(question, k=3)

    context_block = "\n\n".join(
        f"[{h['id']}] {h['title']}\n{h['text']}" for h in hits
    )

    messages = [
        {
            "role": "user",
            "content": (
                f"RETRIEVED_CONTEXT:\n{context_block}\n\n"
                f"QUESTION: {question}"
            ),
        }
    ]

    tool_calls_made = []

    while True:
        response = anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[{
                "name": TOOL_SCHEMA["name"],
                "description": TOOL_SCHEMA["description"],
                "input_schema": TOOL_SCHEMA["input_schema"],
            }],
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return {
                "answer": final_text,
                "retrieved": hits,
                "tool_calls": tool_calls_made,
            }

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = run_tool(block.name, block.input)
            tool_calls_made.append({"tool": block.name, "input": block.input, "result": result})
            if verbose:
                print(f"[tool call] {block.name}({block.input}) -> {result}", file=sys.stderr)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = ask(args.question, verbose=args.verbose)

    print("\n=== ANSWER ===")
    print(result["answer"])

    if args.verbose:
        print("\n=== RETRIEVED CHUNKS ===")
        for h in result["retrieved"]:
            print(f"- [{h['score']:.3f}] {h['title']}")
        print("\n=== TOOL CALLS ===")
        for tc in result["tool_calls"]:
            print(f"- {tc['tool']}({tc['input']}) -> {tc['result']}")


if __name__ == "__main__":
    main()
