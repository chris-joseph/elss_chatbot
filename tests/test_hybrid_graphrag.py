"""
test_hybrid_graphrag.py
Tests for MutualFundKnowledgeGraph, ChromaDenseStore, and HybridRetriever.
"""

import pytest
from rag.knowledge_graph import MutualFundKnowledgeGraph
from rag.chroma_store import ChromaDenseStore
from rag.hybrid_retriever import HybridRetriever

@pytest.fixture
def kg():
    return MutualFundKnowledgeGraph()

@pytest.fixture
def chroma_store():
    return ChromaDenseStore()

@pytest.fixture
def hybrid_retriever():
    return HybridRetriever()

def test_knowledge_graph_structure(kg):
    """Verifies core entity nodes and relations exist in the knowledge graph."""
    assert "Groww ELSS Tax Saver Fund" in kg.graph.nodes
    assert "Section 80C" in kg.graph.nodes
    assert "3 Years" in kg.graph.nodes
    assert "Direct Plan" in kg.graph.nodes
    assert "Systematic Investment Plan (SIP)" in kg.graph.nodes
    
    # Check edges
    assert kg.graph.has_edge("Groww ELSS Tax Saver Fund", "3 Years")
    assert kg.graph.has_edge("Groww ELSS Tax Saver Fund", "NIFTY 500 Total Return Index")
    assert kg.graph.has_edge("Section 80C", "Groww ELSS Tax Saver Fund")

def test_knowledge_graph_multi_hop_query(kg):
    """Tests multi-hop traversal from Section 80C to scheme benchmark."""
    res = kg.query_graph("What is the benchmark of the Section 80C tax saver fund?")
    assert "Groww ELSS Tax Saver Fund" in res["matched_entities"] or "Section 80C" in res["matched_entities"]
    assert "groww_scheme_elss_tax_saver" in res["doc_ids"]

def test_chroma_store_search(chroma_store):
    """Tests semantic search on embedded ChromaDB."""
    assert chroma_store.count() > 0
    res = chroma_store.search("where can I invest for tax savings", top_k=3)
    assert len(res) >= 1
    doc_ids = [p.doc_id for p in res]
    assert any("elss" in d for d in doc_ids)

def test_hybrid_retriever_multi_hop_fusion(hybrid_retriever):
    """Verifies that HybridRetriever fuses graph and lexical streams."""
    res = hybrid_retriever.retrieve("What is the benchmark of the Section 80C tax saver fund?", top_k=2)
    assert len(res) >= 1
    assert "groww_scheme_elss_tax_saver" in res[0].doc_id

def test_hybrid_retriever_out_of_scope_rejection(hybrid_retriever):
    """Verifies that non-financial random tokens are rejected by the hybrid pipeline."""
    res = hybrid_retriever.retrieve("boomba diagga", top_k=2)
    assert len(res) == 0

def test_knowledge_graph_acronym_nodes(kg):
    """Verifies that AUM, CAGR, XIRR, and NAV exist in the knowledge graph."""
    assert "Assets Under Management (AUM)" in kg.graph.nodes
    assert "Compounded Annual Growth Rate (CAGR)" in kg.graph.nodes
    assert "Extended Internal Rate of Return (XIRR)" in kg.graph.nodes
    assert "Net Asset Value (NAV)" in kg.graph.nodes
    res = kg.query_graph("what is AUM in mutual funds")
    assert "Assets Under Management (AUM)" in res["matched_entities"]
    assert "groww_guide_mutual_fund_abbreviations_glossary" in res["doc_ids"]

def test_hybrid_retriever_aum_query(hybrid_retriever):
    """Verifies that HybridRetriever successfully retrieves glossary for AUM."""
    res = hybrid_retriever.retrieve("what is AUM in mutual funds", top_k=2)
    assert len(res) >= 1
    assert "groww_guide_mutual_fund_abbreviations_glossary" in res[0].doc_id

