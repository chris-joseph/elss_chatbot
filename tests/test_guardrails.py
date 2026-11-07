"""
Unit tests for GuardrailsChecker (PII, Advice Refusal, Performance Claims).
"""

import pytest
from rag.guardrails import GuardrailsChecker

@pytest.fixture
def checker():
    return GuardrailsChecker()

def test_pii_pan_detection(checker):
    queries = [
        "My PAN is ABCDE1234F, show my tax report",
        "Here is my pan card: abcde5678g",
        "PAN: BKZPA9988C, can you check my folio?"
    ]
    for q in queries:
        res = checker.check(q)
        assert not res.is_safe
        assert res.status == "refusal_pii"
        assert "security and privacy" in res.message.lower()

def test_pii_aadhaar_detection(checker):
    queries = [
        "Aadhaar: 1234 5678 9012",
        "My aadhar number is 9876 5432 1098"
    ]
    for q in queries:
        res = checker.check(q)
        assert not res.is_safe
        assert res.status == "refusal_pii"

def test_pii_phone_detection(checker):
    queries = [
        "Call me at 9876543210 about ELSS",
        "Contact: +91 9123456789"
    ]
    for q in queries:
        res = checker.check(q)
        assert not res.is_safe
        assert res.status == "refusal_pii"

def test_pii_email_detection(checker):
    queries = [
        "Send my statement to test.investor@gmail.com",
        "My email is mutualfund@sample.org"
    ]
    for q in queries:
        res = checker.check(q)
        assert not res.is_safe
        assert res.status == "refusal_pii"

def test_advice_refusal(checker):
    opinion_queries = [
        "Should I buy Groww ELSS or Groww Multicap?",
        "Which mutual fund scheme is best for me?",
        "Should I invest in Groww Nifty Total Market now?",
        "Recommend me a good mutual fund for 5 years",
        "Is it a good time to invest in banking fund?",
        "Where should I invest my 50000 rupees?"
    ]
    for q in opinion_queries:
        res = checker.check(q)
        assert not res.is_safe
        assert res.status == "refusal_advice"
        assert "do not provide investment advice" in res.message
        assert res.citation_url == "https://www.amfiindia.com/investor"

def test_performance_claim_refusal(checker):
    performance_queries = [
        "How much return will Groww Multicap give in 3 years?",
        "Which fund gives higher return, ELSS or Nifty?",
        "Will Groww Nifty Total Market give 15% return?",
        "Calculate my future returns if I invest 5000 per month"
    ]
    for q in performance_queries:
        res = checker.check(q)
        assert not res.is_safe
        assert res.status == "refusal_performance"
        assert "cannot compute, project, or compare fund returns" in res.message
        assert res.citation_url == "https://www.growwmf.in/downloads/factsheets"

def test_safe_factual_queries_pass(checker):
    safe_queries = [
        "What is the expense ratio of Groww ELSS Tax Saver Fund?",
        "What is the exit load of Groww Liquid Fund?",
        "What is the mandatory lock-in period for ELSS mutual funds?",
        "How can I download my capital gains statement on Groww?",
        "What is the minimum SIP investment for Groww Multicap Fund?",
        "What is the benchmark of Groww Nifty Total Market Index Fund?",
        "What are SEBI cut-off timings for mutual funds?"
    ]
    for q in safe_queries:
        res = checker.check(q)
        assert res.is_safe
        assert res.status == "ok"
