"""
Extended unit tests for GuardrailsChecker covering all regex patterns and edge cases.
"""

import pytest
from rag.guardrails import GuardrailsChecker

@pytest.fixture
def checker():
    return GuardrailsChecker()

def test_otp_and_account_number_blocking(checker):
    queries = [
        "My OTP is 123456, please process",
        "Here is the one time password 9876",
        "My account number: 12345678901234",
        "Bank account: 987654321098"
    ]
    for q in queries:
        res = checker.check(q)
        assert not res.is_safe
        assert res.status == "refusal_pii"
        assert "sensitive financial credentials" in res.message.lower()

def test_extended_advice_patterns(checker):
    advice_queries = [
        "Where should I put my savings?",
        "How should I allocate my monthly income?",
        "Help me choose an index fund",
        "Help me pick between multicap and liquid",
        "What is your recommendation for 2026?",
        "What is your advice on mutual funds?",
        "Will I become rich by investing in Groww?",
        "Is Groww ELSS better than PPF?",
        "Which one should I select?"
    ]
    for q in advice_queries:
        res = checker.check(q)
        assert not res.is_safe
        assert res.status == "refusal_advice"
        assert "do not provide investment advice" in res.message

def test_extended_performance_patterns(checker):
    perf_queries = [
        "What CAGR will I get in 5 years?",
        "How much profit will this fund yield?",
        "What are the projected returns for this year?",
        "What are the expected returns of Groww Liquid Fund?",
        "Will this scheme give 12% returns?"
    ]
    for q in perf_queries:
        res = checker.check(q)
        assert not res.is_safe
        assert res.status == "refusal_performance"
        assert "cannot compute, project, or compare fund returns" in res.message

def test_guardrails_edge_cases(checker):
    # Whitespace query
    res = checker.check("   ")
    assert res.is_safe
    assert res.status == "ok"
