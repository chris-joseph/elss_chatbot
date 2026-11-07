"""
Unit tests for SourceRetriever.
"""

import os
import pytest
from rag.retriever import SourceRetriever

@pytest.fixture
def retriever():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sources_dir = os.path.join(base_dir, "sources")
    return SourceRetriever(sources_dir=sources_dir)

def test_retriever_initialization(retriever):
    assert retriever.total_passages > 10
    assert len(retriever.doc_freq) > 50

def test_retrieve_elss_scheme(retriever):
    passages = retriever.retrieve("What is the expense ratio and exit load of Groww ELSS Tax Saver Fund?", top_k=3)
    assert len(passages) > 0
    top_doc_ids = [p.doc_id for p in passages]
    assert "groww_scheme_elss_tax_saver" in top_doc_ids
    assert "groww.in" in passages[0].url

def test_retrieve_capital_gains(retriever):
    passages = retriever.retrieve("How to download capital gains statement on Groww?", top_k=3)
    assert len(passages) > 0
    top_doc_ids = [p.doc_id for p in passages]
    assert any("capital_gains" in did for did in top_doc_ids)

def test_retrieve_liquid_exit_load(retriever):
    passages = retriever.retrieve("Exit load of Groww Liquid Fund", top_k=3)
    assert len(passages) > 0
    top_doc_ids = [p.doc_id for p in passages]
    assert "groww_scheme_liquid_fund" in top_doc_ids

def test_retrieve_cas_statement(retriever):
    passages = retriever.retrieve("How can I download Consolidated Account Statement CAS?", top_k=3)
    assert len(passages) > 0
    top_urls = [p.url for p in passages]
    assert any("amfiindia" in u or "sebi.gov.in" in u for u in top_urls)

def test_retrieve_mutual_fund_definition(retriever):
    passages = retriever.retrieve("what is a Mutal fund", top_k=3)
    assert len(passages) > 0
    top_p = passages[0]
    assert "what is a mutual fund" in top_p.title.lower()
    assert "expense-ratio" not in top_p.url

def test_retrieve_nav_definition(retriever):
    passages = retriever.retrieve("what is nav", top_k=3)
    assert len(passages) > 0
    assert "nav" in passages[0].title.lower() or "net asset value" in passages[0].title.lower()

def test_retrieve_riskometer_definition(retriever):
    passages = retriever.retrieve("what is a riskometer", top_k=3)
    assert len(passages) > 0
    assert "riskometer" in passages[0].title.lower()
