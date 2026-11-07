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
    assert len(sentences) <= 10

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

def test_engine_amc_definition(engine):
    """Verifies that 'what is AMC' retrieves the AMC guide, not demat charges."""
    res = engine.answer("what is AMC")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "asset management company" in ans or "amc" in ans
    assert "https://groww.in/p/what-is-amc" in res["answer"]

def test_engine_abbreviation_ter(engine):
    """Verifies that 'what does TER mean' correctly answers with Total Expense Ratio."""
    res = engine.answer("what does TER mean")
    assert res["status"] == "factual"
    assert "ter" in res["answer"].lower() or "expense ratio" in res["answer"].lower()

def test_engine_common_indices(engine):
    """Verifies that 'what are common mutual fund indices' retrieves benchmark indices."""
    res = engine.answer("what are common mutual fund indices")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "nifty 50" in ans or "sensex" in ans or "benchmark" in ans
    assert "https://groww.in/p/benchmark" in res["answer"]

def test_engine_idcw_definition(engine):
    """Verifies that 'what is IDCW' retrieves the growth vs IDCW guide."""
    res = engine.answer("what is IDCW")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "income distribution" in ans or "idcw" in ans
    assert "https://groww.in/p/growth-vs-idcw-options" in res["answer"]

def test_engine_aum_definition(engine):
    """Verifies that 'what is AUM in mutual funds' retrieves the acronym glossary definition."""
    res = engine.answer("what is AUM in mutual funds")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "assets under management" in ans or "aum" in ans

def test_engine_cagr_xirr(engine):
    """Verifies that 'what is CAGR and XIRR' returns definition of return metrics."""
    res = engine.answer("what is CAGR and XIRR")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "cagr" in ans or "growth rate" in ans or "xirr" in ans


def test_engine_stocks_vs_mutual_funds(engine):
    """Verifies that mutual funds vs stocks query retrieves the asset comparison guide."""
    res = engine.answer("difference between mutual funds and stocks")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "stock" in ans
    assert "https://groww.in/p/mutual-funds-vs-stocks-fd" in res["answer"]

def test_engine_out_of_scope_rejection(engine):
    """Verifies that random gibberish or non-mutual fund queries are rejected."""
    r1 = engine.answer("boomba diagga")
    assert r1["status"] == "out_of_scope"
    assert "No verified official sources were found" in r1["answer"]

    r2 = engine.answer("what time is it now")
    assert r2["status"] == "out_of_scope"
    assert "No verified official sources were found" in r2["answer"]

def test_engine_vs_direct_vs_sip(engine):
    """Verifies that 'difference b/w direct and sip' provides a comprehensive multi-sentence comparison."""
    res = engine.answer("what is the difference b/w direct and sip")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "direct" in ans
    assert "sip" in ans
    # Check completeness: should explain both concepts within 10 sentences
    main_text = res["answer"].split("Last updated from sources:")[0].strip()
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', main_text) if s.strip()]
    assert 2 <= len(sentences) <= 10

def test_engine_vs_mf_vs_fd(engine):
    """Verifies that shorthand 'mf vs fd' correctly retrieves comparison and produces factual answer."""
    res = engine.answer("mf vs fd")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "fd" in ans or "fixed deposit" in ans
    assert "https://groww.in/p/mutual-funds-vs-stocks-fd" in res["answer"]

def test_engine_vs_direct_vs_regular(engine):
    """Verifies 'direct vs regular' and 'direct v/s regular' both succeed."""
    r1 = engine.answer("direct vs regular")
    r2 = engine.answer("direct v/s regular")
    assert r1["status"] == "factual"
    assert r2["status"] == "factual"
    assert "https://groww.in/p/direct-vs-regular-mutual-funds" in r1["answer"]
    assert "https://groww.in/p/direct-vs-regular-mutual-funds" in r2["answer"]


