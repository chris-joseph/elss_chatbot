"""
test_out_of_scope.py
Unit and integration tests for Out-of-Domain (OOD) and unverified query rejection.
Ensures nonsense, chit-chat, and general world knowledge queries are rejected without
degrading in-domain factual retrieval accuracy.
"""

import pytest
from rag.engine import RAGEngine
from rag.retriever import SourceRetriever

@pytest.fixture
def engine():
    return RAGEngine()

@pytest.fixture
def retriever():
    return SourceRetriever()

def test_gibberish_queries_rejected(engine):
    gibberish_samples = [
        "boomba diagga?",
        "asdfghjkl qwerty uiop",
        "blabla foo bar baz 123",
        "xyz987 nonsense token"
    ]
    for q in gibberish_samples:
        res = engine.answer(q)
        assert res["status"] == "out_of_scope", f"Query '{q}' should be out_of_scope, got {res['status']}"
        assert "no verified official sources" in res["answer"].lower()
        assert "growwmf.in" in res["source_url"]

def test_unrelated_world_knowledge_queries_rejected(engine):
    unrelated_samples = [
        "what time is it now?",
        "who is the prime minister of india?",
        "what is the recipe for pasta?",
        "how is the weather in bangalore today?",
        "tell me a joke",
        "who won the cricket world cup?",
        "write python code for binary search"
    ]
    for q in unrelated_samples:
        res = engine.answer(q)
        assert res["status"] == "out_of_scope", f"Query '{q}' should be out_of_scope, got {res['status']}"
        assert "no verified official sources" in res["answer"].lower()

def test_in_domain_queries_retained(engine):
    in_domain_samples = [
        "what is a Mutal fund",
        "What is the expense ratio and exit load of Groww ELSS Tax Saver Fund?",
        "what is nav",
        "what is a riskometer",
        "how can i cancel my sip on groww",
        "what is the lock-in period for ELSS mutual funds?",
        "exit load of Groww Liquid Fund",
        "minimum sip for Groww Multicap Fund",
        "benchmark of Groww Nifty Total Market Index Fund",
        "how to download capital gains statement on groww"
    ]
    for q in in_domain_samples:
        res = engine.answer(q)
        assert res["status"] == "factual", f"In-domain query '{q}' was incorrectly marked as {res['status']}"
        assert len(res["answer"]) > 20
        assert "Last updated from sources:" in res["answer"]
        assert res["source_url"] is not None

def test_retriever_score_threshold_filtering(retriever):
    # Out of scope queries should produce empty list
    assert len(retriever.retrieve("boomba diagga?")) == 0
    assert len(retriever.retrieve("what time is it now?")) == 0
    assert len(retriever.retrieve("recipe for pasta")) == 0

    # In domain queries should produce top passages
    elss_passages = retriever.retrieve("Groww ELSS Tax Saver Fund expense ratio")
    assert len(elss_passages) > 0
    assert elss_passages[0].score >= 0.45

    mf_passages = retriever.retrieve("what is a Mutal fund")
    assert len(mf_passages) > 0
    assert mf_passages[0].score >= 0.45
