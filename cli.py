"""
cli.py
Command-Line Interface and Headless Evaluator for Groww Mutual Fund FAQ Assistant.
"""

import sys
import argparse
from typing import Optional
from rag.engine import RAGEngine

SAMPLE_EVAL_QUERIES = [
    "What is the expense ratio and exit load of Groww ELSS Tax Saver Fund?",
    "What is the mandatory lock-in period for ELSS mutual funds?",
    "How can I download my Mutual Fund capital gains statement on Groww?",
    "What is the minimum SIP investment for Groww Multicap Fund?",
    "What is the exit load schedule for Groww Liquid Fund?",
    "What is the benchmark index for Groww Nifty Total Market Index Fund?",
    "What are the official SEBI cut-off timings for equity and liquid mutual funds?",
    "Should I buy Groww ELSS or Groww Multicap for the long term?",
    "My PAN is ABCDE1234F, show me my tax statement.",
    "Will Groww Nifty Total Market Index Fund give 15% annual return?"
]

def run_eval(engine: RAGEngine):
    status = engine.get_llm_status()
    print("=" * 80)
    print(" GROWW MUTUAL FUND FACTS-ONLY FAQ ASSISTANT - HEADLESS EVALUATION")
    print(f" LLM Status: {status['status_label']}")
    print("=" * 80)
    print()

    for i, q in enumerate(SAMPLE_EVAL_QUERIES, 1):
        print(f"[{i}] Question: {q}")
        res = engine.answer(q)
        print(f"    Status: {res['status']} | Latency: {res['latency_ms']} ms")
        print(f"    Answer: {res['answer']}")
        if res.get('source_url'):
            print(f"    Source Link: {res['source_url']}")
        print("-" * 80)
        print()

def run_single_query(engine: RAGEngine, query: str):
    res = engine.answer(query)
    print(f"[{res['status'].upper()}]")
    print(res['answer'])
    if res.get('source_url'):
        print(f"Source: {res['source_url']}")

def run_interactive(engine: RAGEngine):
    status = engine.get_llm_status()
    print("=" * 70)
    print(" Groww Mutual Fund FAQ Assistant (CLI Mode)")
    print(f" Status: {status['status_label']}")
    print(" Facts-only. No investment advice.")
    print(" Type 'quit' or 'exit' to leave.")
    print("=" * 70)
    print()

    while True:
        try:
            query = input("Ask a question: ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            res = engine.answer(query)
            print()
            print(f"[{res['status'].upper()}]")
            print(res['answer'])
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

def main():
    parser = argparse.ArgumentParser(description="Groww Mutual Fund FAQ Assistant CLI")
    parser.add_argument("--eval", action="store_true", help="Run automated evaluation against sample queries")
    parser.add_argument("--query", "-q", type=str, default=None, help="Ask a single question and print response")
    parser.add_argument("--provider", type=str, default=None, help="LLM Provider (groq, deepseek, openai, ollama, offline)")
    parser.add_argument("--model", type=str, default=None, help="Model identifier")
    parser.add_argument("--api-key", type=str, default=None, help="API key for selected provider")
    args = parser.parse_args()

    engine = RAGEngine(
        provider=args.provider,
        api_key=args.api_key,
        model=args.model
    )

    if args.query:
        run_single_query(engine, args.query)
    elif args.eval:
        run_eval(engine)
    else:
        run_interactive(engine)

if __name__ == "__main__":
    main()
