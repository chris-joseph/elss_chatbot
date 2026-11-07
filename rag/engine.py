"""
engine.py
Unified RAG Engine for Groww Mutual Fund FAQ Assistant.
Coordinates Guardrails, Source Retrieval, and Generator execution.
"""

import time
from typing import Dict, Any, Optional
from rag.guardrails import GuardrailsChecker
from rag.retriever import SourceRetriever
from rag.generator import FactsOnlyGenerator

class RAGEngine:
    def __init__(self, sources_dir: Optional[str] = None):
        self.guardrails = GuardrailsChecker()
        self.retriever = SourceRetriever(sources_dir=sources_dir)
        self.generator = FactsOnlyGenerator()

    def answer(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        normalized_query = query.strip()

        if not normalized_query:
            return {
                "answer": "Please enter a factual question regarding mutual funds, such as expense ratio, exit load, minimum SIP, lock-in, or downloading statements.",
                "status": "empty_query",
                "source_url": None,
                "source_title": None,
                "latency_ms": 0.0
            }

        # Step 1: Run Guardrails
        guardrail_res = self.guardrails.check(normalized_query)
        if not guardrail_res.is_safe:
            latency = (time.time() - start_time) * 1000.0
            return {
                "answer": guardrail_res.message,
                "status": guardrail_res.status,
                "source_url": guardrail_res.citation_url,
                "source_title": guardrail_res.citation_title,
                "latency_ms": round(latency, 2)
            }

        # Step 2: Retrieve Relevant Passages
        passages = self.retriever.retrieve(normalized_query, top_k=3)

        # Step 3: Generate Grounded Facts-Only Answer
        gen_result = self.generator.generate(normalized_query, passages)
        latency = (time.time() - start_time) * 1000.0

        return {
            "answer": gen_result["answer"],
            "status": "factual",
            "source_url": gen_result["source_url"],
            "source_title": gen_result["source_title"],
            "latency_ms": round(latency, 2),
            "retrieved_passages_count": len(passages)
        }
