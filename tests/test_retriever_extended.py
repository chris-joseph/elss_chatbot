"""
Extended unit tests for SourceRetriever covering error boundaries, empty dirs, and properties.
"""

import os
import tempfile
import pytest
from rag.retriever import SourceRetriever

def test_missing_directory_handling():
    retriever = SourceRetriever(sources_dir="/non/existent/path/for/tests")
    assert retriever.total_passages == 0
    assert retriever.doc_freq == {}
    assert retriever.retrieve("any query") == []

def test_empty_directory_handling():
    with tempfile.TemporaryDirectory() as tmp_dir:
        retriever = SourceRetriever(sources_dir=tmp_dir)
        assert retriever.total_passages == 0
        assert retriever.doc_freq == {}
        assert retriever.retrieve("query") == []

def test_parse_frontmatter_edge_cases():
    retriever = SourceRetriever(sources_dir="/non/existent/path")
    
    # Text without frontmatter
    meta, body = retriever._parse_frontmatter("Just plain markdown text")
    assert meta == {}
    assert body == "Just plain markdown text"
    
    # Incomplete frontmatter
    meta, body = retriever._parse_frontmatter("--- title: incomplete")
    assert meta == {}
    
    # Valid frontmatter with quotes
    valid_text = '---\nid: test_doc\ntitle: "Test Title"\n---\nBody here'
    meta, body = retriever._parse_frontmatter(valid_text)
    assert meta["id"] == "test_doc"
    assert meta["title"] == "Test Title"
    assert body == "Body here"

def test_retriever_top_k_boundaries():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sources_dir = os.path.join(base_dir, "sources")
    retriever = SourceRetriever(sources_dir=sources_dir)
    
    # top_k = 1
    p1 = retriever.retrieve("Groww ELSS", top_k=1)
    assert len(p1) == 1
    
    # top_k = 5
    p5 = retriever.retrieve("Groww Mutual Fund", top_k=5)
    assert len(p5) == 5

def test_retriever_properties():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sources_dir = os.path.join(base_dir, "sources")
    retriever = SourceRetriever(sources_dir=sources_dir)
    
    assert retriever.total_passages > 0
    assert isinstance(retriever.doc_freq, dict)
    assert len(retriever.doc_freq) > 0
