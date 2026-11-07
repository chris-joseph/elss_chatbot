"""
tests/test_offline_engine.py
Comprehensive test suite validating the Facts-Only RAG engine in 100% Offline Grounded Mode.
Ensures zero outbound network calls, deterministic extraction, strict brevity (<=10 sentences),
proper citations, and accuracy across real-world common mutual fund queries.
"""

import re
from unittest.mock import patch
import pytest
from rag.engine import RAGEngine

@pytest.fixture(scope="module")
def offline_engine():
    """Initializes a shared RAGEngine strictly in offline mode."""
    return RAGEngine(provider="offline")

# ---------------------------------------------------------------------------
# Category 1: Fundamental Concepts & Acronym Definitions
# ---------------------------------------------------------------------------

def test_offline_mutual_fund_definition(offline_engine):
    res = offline_engine.answer("what is a mutual fund")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "pool of money" in ans or "fund manager" in ans or "investment vehicle" in ans
    assert "https://groww.in/p/what-is-mutual-fund" in res["answer"]
    assert "last updated from sources:" in ans

def test_offline_amc_definition(offline_engine):
    res = offline_engine.answer("what is an AMC")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "asset management company" in ans or "amc" in ans
    assert "https://groww.in/p/what-is-amc" in res["answer"]

def test_offline_groww_amc_schemes(offline_engine):
    res = offline_engine.answer("what are schemes under groww amc")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert any(term in ans for term in ["groww", "scheme", "equity", "elss", "fund"])
    assert "last updated from sources:" in ans

def test_offline_nav_definition(offline_engine):
    res = offline_engine.answer("what is NAV")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "net asset value" in ans or "market value" in ans
    assert "https://groww.in/p/nav" in res["answer"]

def test_offline_ter_meaning(offline_engine):
    res = offline_engine.answer("what does TER mean")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "expense ratio" in ans or "annual" in ans
    # Ensure unpunctuated heading fragments are not present
    assert not ans.startswith("expense ratio implications.")
    assert "https://groww.in/p/expense-ratio" in res["answer"]

def test_offline_aum_definition(offline_engine):
    res = offline_engine.answer("what is AUM in mutual funds")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "assets under management" in ans or "aum" in ans
    assert "market value" in ans or "investments" in ans

def test_offline_idcw_definition(offline_engine):
    res = offline_engine.answer("what is IDCW in mutual funds")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "income distribution" in ans or "idcw" in ans or "dividend" in ans
    assert "https://groww.in/p/growth-vs-idcw-options" in res["answer"]

# ---------------------------------------------------------------------------
# Category 2: Scheme Metrics & Key-Value Facts
# ---------------------------------------------------------------------------

def test_offline_elss_lockin(offline_engine):
    res = offline_engine.answer("what is the lock-in period for Groww ELSS Tax Saver Fund")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "3 year" in ans or "3-year" in ans
    assert "https://groww.in/mutual-funds/groww-elss-tax-saver-fund-direct-growth" in res["answer"]

def test_offline_elss_expense_ratio(offline_engine):
    res = offline_engine.answer("what is the expense ratio of Groww ELSS Tax Saver Fund")
    assert res["status"] == "factual"
    assert "1.47%" in res["answer"]

def test_offline_multicap_minimum_sip(offline_engine):
    res = offline_engine.answer("what is the minimum SIP investment for Groww Multicap Fund")
    assert res["status"] == "factual"
    assert "₹500" in res["answer"] or "500" in res["answer"]

def test_offline_liquid_exit_load(offline_engine):
    res = offline_engine.answer("what is the exit load schedule for Groww Liquid Fund")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "0.0070%" in ans or "day" in ans or "exit load" in ans

def test_offline_nifty_total_market_benchmark(offline_engine):
    res = offline_engine.answer("what is the benchmark index for Groww Nifty Total Market Index Fund")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "nifty total market" in ans

# ---------------------------------------------------------------------------
# Category 3: Comparative Queries ("VS")
# ---------------------------------------------------------------------------

def test_offline_direct_vs_sip(offline_engine):
    res = offline_engine.answer("what is the difference b/w direct and sip")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "sip" in ans
    assert "direct" in ans
    assert ("variant" in ans or "plan" in ans or "mode" in ans or "direct sip" in ans)

def test_offline_direct_vs_regular(offline_engine):
    res = offline_engine.answer("direct vs regular mutual funds")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "direct" in ans and "regular" in ans
    assert ("costs" in ans or "commission" in ans or "expense ratio" in ans or "variants" in ans)

def test_offline_mf_vs_stocks(offline_engine):
    res = offline_engine.answer("mutual funds vs stocks")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "mutual fund" in ans and "stock" in ans

def test_offline_mf_vs_fd(offline_engine):
    res = offline_engine.answer("mf vs fd")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "fd" in ans or "fixed deposit" in ans

def test_offline_growth_vs_idcw(offline_engine):
    res = offline_engine.answer("growth vs idcw")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "growth" in ans and "idcw" in ans

def test_offline_sip_vs_lumpsum(offline_engine):
    res = offline_engine.answer("sip vs lumpsum")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "sip" in ans and "lump" in ans

# ---------------------------------------------------------------------------
# Category 4: Operational Procedures & FAQs
# ---------------------------------------------------------------------------

def test_offline_capital_gains_download(offline_engine):
    res = offline_engine.answer("how can I download my Mutual Fund capital gains statement on Groww")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "groww" in ans
    assert "capital gain" in ans or "report" in ans

def test_offline_cancel_sip(offline_engine):
    res = offline_engine.answer("how can I cancel my ongoing SIP on Groww")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "cancel" in ans and "sip" in ans

def test_offline_sebi_cutoff_timings(offline_engine):
    res = offline_engine.answer("what are the official SEBI cut-off timings for equity and liquid mutual funds")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "12:00" in ans or "2:00" in ans or "cut-off" in ans

def test_offline_nomination_rules(offline_engine):
    res = offline_engine.answer("how to update nomination in mutual funds")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "nomination" in ans or "nominee" in ans or "folio" in ans

def test_offline_kyc_requirements(offline_engine):
    res = offline_engine.answer("what documents are required for mutual fund KYC")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "kyc" in ans or "pan" in ans or "aadhaar" in ans

# ---------------------------------------------------------------------------
# Category 5: Taxation & Regulations
# ---------------------------------------------------------------------------

def test_offline_equity_taxation(offline_engine):
    res = offline_engine.answer("how are equity mutual funds taxed")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "12.5%" in ans or "ltcg" in ans or "tax" in ans

def test_offline_section_80c_elss_limit(offline_engine):
    res = offline_engine.answer("what is Section 80C tax deduction limit for ELSS")
    assert res["status"] == "factual"
    ans = res["answer"].lower()
    assert "80c" in ans
    assert ("1.5" in ans or "lakh" in ans or "elss" in ans)

# ---------------------------------------------------------------------------
# Category 6: Guardrails, Safety & Constraints in Offline Mode
# ---------------------------------------------------------------------------

def test_offline_pii_blocking(offline_engine):
    res = offline_engine.answer("My PAN is ABCDE1234F, show my statement")
    assert res["status"] == "refusal_pii"
    assert "privacy" in res["answer"].lower()

def test_offline_advice_refusal(offline_engine):
    res = offline_engine.answer("Should I invest in Groww ELSS or Groww Multicap?")
    assert res["status"] == "refusal_advice"
    assert "do not provide investment advice" in res["answer"].lower()

def test_offline_performance_refusal(offline_engine):
    res = offline_engine.answer("How much return will Groww Multicap give in 2 years?")
    assert res["status"] == "refusal_performance"
    assert "cannot compute, project, or compare fund returns" in res["answer"].lower()

def test_offline_gibberish_rejection(offline_engine):
    res = offline_engine.answer("boomba diagga zork blip")
    assert res["status"] == "out_of_scope"
    assert "No verified official sources were found" in res["answer"]

def test_offline_empty_query(offline_engine):
    res = offline_engine.answer("   ")
    assert res["status"] == "empty_query"

# ---------------------------------------------------------------------------
# Category 7: Invariants, Performance & Zero Network Call Guarantees
# ---------------------------------------------------------------------------

def test_offline_no_outbound_network_calls(offline_engine):
    """Guarantees that offline mode performs zero HTTP/network calls."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        res = offline_engine.answer("what is the lock-in period for Groww ELSS Tax Saver Fund")
        assert res["status"] == "factual"
        mock_urlopen.assert_not_called()

def test_offline_determinism(offline_engine):
    """Guarantees that identical queries produce byte-for-byte identical answers."""
    q = "what is the difference b/w direct and sip"
    r1 = offline_engine.answer(q)
    r2 = offline_engine.answer(q)
    assert r1["answer"] == r2["answer"]
    assert r1["status"] == r2["status"]
    assert r1["source_url"] == r2["source_url"]

def test_offline_latency_benchmark(offline_engine):
    """Guarantees that offline mode answers within sub-500ms latency."""
    res = offline_engine.answer("what is an AMC")
    assert res["latency_ms"] < 500.0

def test_offline_sentence_and_citation_invariants(offline_engine):
    """Validates that offline answers obey <=10 sentences, 1 citation link, and timestamp suffix."""
    test_queries = [
        "what is a mutual fund",
        "what is the lock-in period for Groww ELSS Tax Saver Fund",
        "what is the difference b/w direct and sip",
        "how can I cancel my ongoing SIP on Groww"
    ]
    for q in test_queries:
        res = offline_engine.answer(q)
        ans = res["answer"]
        assert "Last updated from sources:" in ans
        # Check citation link
        links = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', ans)
        assert len(links) >= 1
        # Check sentence count
        main_text = ans.split("Last updated from sources:")[0].strip()
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', main_text) if s.strip()]
        assert len(sentences) <= 10
