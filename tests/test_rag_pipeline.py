"""
End-to-end tests for RAGEngine.
Validates answer structure, constraint compliance, citation links, and brevity.
"""

import re
import pytest
from rag.engine import RAGEngine

@pytest.fixture
def engine():
    return RAGEngine()

def test_engine_empty_query(engine):
    res = engine.answer("")
    assert res["status"] == "empty_query"

def test_engine_pii_blocking(engine):
    res = engine.answer("My PAN is ABCDE1234F, tell me my tax status")
    assert res["status"] == "refusal_pii"
    assert "security and privacy" in res["answer"].lower()

def test_engine_advice_refusal(engine):
    res = engine.answer("Should I invest in Groww ELSS or Groww Nifty Total Market?")
    assert res["status"] == "refusal_advice"
    assert "do not provide investment advice" in res["answer"].lower()
    assert "https://www.amfiindia.com/investor" in res["answer"]

def test_engine_performance_refusal(engine):
    res = engine.answer("How much return will Groww Multicap give in 2 years?")
    assert res["status"] == "refusal_performance"
    assert "cannot compute, project, or compare fund returns" in res["answer"].lower()
    assert "https://www.growwmf.in/downloads/factsheets" in res["answer"]

def test_engine_factual_elss_lockin(engine):
    res = engine.answer("What is the lock-in period for Groww ELSS Tax Saver Fund?")
    assert res["status"] == "factual"
    ans = res["answer"]
    assert "3 year" in ans.lower() or "3-year" in ans.lower()
    links = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', ans)
    assert len(links) >= 1
    assert "Last updated from sources:" in ans
    main_text = ans.split("Last updated from sources:")[0].strip()
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', main_text) if s.strip()]
    assert len(sentences) <= 3

def test_engine_factual_capital_gains(engine):
    res = engine.answer("How do I download my mutual fund capital gains statement on Groww?")
    assert res["status"] == "factual"
    ans = res["answer"]
    assert "groww" in ans.lower()
    assert "report" in ans.lower() or "capital gain" in ans.lower()
    assert "Last updated from sources:" in ans

def test_engine_factual_liquid_exit_load(engine):
    res = engine.answer("What is the exit load of Groww Liquid Fund?")
    assert res["status"] == "factual"
    ans = res["answer"]
    assert "0.007" in ans or "exit load" in ans.lower() or "day" in ans.lower()
    assert "Last updated from sources:" in ans

def test_engine_distinct_queries_produce_distinct_responses(engine):
    """Ensures sequential diverse queries do not yield static/identical answers."""
    q1 = "What is the expense ratio of Groww ELSS Tax Saver Fund?"
    q2 = "How can I cancel my ongoing SIP on Groww?"
    q3 = "What is the benchmark of Groww Banking & Financial Services Fund?"

    r1 = engine.answer(q1)
    r2 = engine.answer(q2)
    r3 = engine.answer(q3)

    assert r1["status"] == "factual"
    assert r2["status"] == "factual"
    assert r3["status"] == "factual"

    # Answers must be distinctly different
    assert r1["answer"] != r2["answer"]
    assert r2["answer"] != r3["answer"]
    assert r1["answer"] != r3["answer"]

    # Verify specific contents
    assert "1.47%" in r1["answer"]
    assert "cancel" in r2["answer"].lower()
    assert "financial services" in r3["answer"].lower()
