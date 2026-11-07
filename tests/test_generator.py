"""
Unit tests for FactsOnlyGenerator covering sentence truncation, citation insertion,
all deterministic extraction branches, and LLM orchestration with mocks.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from rag.generator import FactsOnlyGenerator
from rag.retriever import Passage

@pytest.fixture
def sample_passage():
    return Passage(
        doc_id="groww_scheme_elss_tax_saver",
        title="Groww ELSS Tax Saver Fund - Scheme Facts & Details",
        url="https://groww.in/mutual-funds/groww-elss-tax-saver-fund-direct-growth",
        organization="Groww Mutual Fund",
        category="ELSS / Tax Saver",
        last_updated="2026-11-07",
        section_title="Official Scheme Overview & Key Facts",
        content=(
            "- **Fund House / AMC:** Groww Mutual Fund\n"
            "- **Expense Ratio (Direct Growth):** 1.47%\n"
            "- **Exit Load:** Nil\n"
            "- **Statutory Lock-in Period:** 3 Years\n"
            "- **Minimum SIP Investment:** ₹500\n"
            "- **Minimum Lump-sum Investment:** ₹500\n"
            "- **Benchmark:** NIFTY 500 Total Return Index"
        )
    )

def test_truncate_to_sentences():
    gen = FactsOnlyGenerator()
    
    # 2 sentences should remain unchanged
    text_2 = "First sentence. Second sentence. Last updated from sources: 2026-11-07"
    assert gen._truncate_to_sentences(text_2, max_sentences=3) == text_2
    
    # 5 sentences should be truncated to 3
    text_5 = "One. Two. Three. Four. Five. Last updated from sources: 2026-11-07"
    res = gen._truncate_to_sentences(text_5, max_sentences=3)
    assert res == "One. Two. Three. Last updated from sources: 2026-11-07"
    
    # Text without last updated
    text_no_update = "Sentence 1. Sentence 2. Sentence 3. Sentence 4."
    res_no_update = gen._truncate_to_sentences(text_no_update, max_sentences=3)
    assert res_no_update == "Sentence 1. Sentence 2. Sentence 3."

def test_ensure_citation(sample_passage):
    gen = FactsOnlyGenerator()
    
    # Text already with markdown citation
    text_with_link = "Here is fact [Groww ELSS](https://groww.in/mutual-funds) Last updated from sources: 2026-11-07"
    assert gen._ensure_citation(text_with_link, sample_passage) == text_with_link
    
    # Text without citation link
    text_without_link = "This is a fact. Last updated from sources: 2026-11-07"
    ensured = gen._ensure_citation(text_without_link, sample_passage)
    assert "https://groww.in/mutual-funds/groww-elss-tax-saver-fund-direct-growth" in ensured
    assert "Last updated from sources: 2026-11-07" in ensured
    
    # Text without timestamp and without link
    plain_text = "Simple statement."
    ensured_plain = gen._ensure_citation(plain_text, sample_passage)
    assert "https://groww.in/mutual-funds/groww-elss-tax-saver-fund-direct-growth" in ensured_plain

def test_deterministic_extract_branches(sample_passage):
    gen = FactsOnlyGenerator()
    passages = [sample_passage]
    
    # Cancel SIP
    ans = gen._deterministic_extract("how do i cancel my sip on groww?", passages)
    assert "cancel" in ans.lower()
    assert "7 days" in ans
    
    # Start SIP
    ans = gen._deterministic_extract("how to start a sip?", passages)
    assert "start an sip" in ans.lower() or "autopay" in ans.lower()
    
    # Expense Ratio
    ans = gen._deterministic_extract("what is the expense ratio?", passages)
    assert "1.47%" in ans
    
    # Exit Load
    ans = gen._deterministic_extract("what is the exit load?", passages)
    assert "nil" in ans.lower()
    
    # Lock-in
    ans = gen._deterministic_extract("is there a lock-in period for elss?", passages)
    assert "3 years" in ans
    
    # Minimum SIP
    ans = gen._deterministic_extract("what is the minimum sip amount?", passages)
    assert "₹500" in ans
    
    # Capital Gains
    ans = gen._deterministic_extract("download capital gains statement", passages)
    assert "capital gain" in ans.lower()
    
    # CAS Statement
    ans = gen._deterministic_extract("how to get my cas statement?", passages)
    assert "consolidated account statement" in ans.lower()
    
    # Benchmark
    ans = gen._deterministic_extract("what is the benchmark index?", passages)
    assert "nifty 500" in ans.lower()
    
    # Cut-off timings
    ans = gen._deterministic_extract("what are the cut-off timings?", passages)
    assert "2:00 pm" in ans.lower()

    # Mutual Fund definition (including typo handling)
    ans = gen._deterministic_extract("what is a Mutal fund", passages)
    assert "pools capital" in ans.lower() or "pool" in ans.lower()
    assert "expense ratio" not in ans.lower()

    # NAV definition
    ans = gen._deterministic_extract("what is nav", passages)
    assert "market value per unit" in ans.lower()

    # Riskometer definition
    ans = gen._deterministic_extract("what is a riskometer", passages)
    assert "graphical representation" in ans.lower()
    
    # Generic summary fallback
    ans = gen._deterministic_extract("tell me about the fund manager", passages)
    assert "official details" in ans.lower()

def test_generate_empty_passages():
    gen = FactsOnlyGenerator()
    res = gen.generate("some question", [])
    assert "No verified official sources" in res["answer"]
    assert res["source_url"] == "https://www.growwmf.in/downloads/factsheets"

def test_call_llm_without_key():
    gen = FactsOnlyGenerator(api_key="")
    assert gen._call_llm("test prompt", "test system") is None

def test_call_llm_mock_success(sample_passage):
    gen = FactsOnlyGenerator(api_key="test-api-key")
    mock_response_json = {
        "choices": [
            {
                "message": {
                    "content": "The expense ratio is 1.47%. [Groww ELSS](https://groww.in). Last updated from sources: 2026-11-07"
                }
            }
        ]
    }
    
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = json.dumps(mock_response_json).encode("utf-8")
    
    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = gen._call_llm("prompt", "system")
        assert res is not None
        assert "1.47%" in res

def test_call_llm_mock_exception(sample_passage):
    gen = FactsOnlyGenerator(api_key="test-api-key")
    with patch("urllib.request.urlopen", side_effect=Exception("Connection timed out")):
        res = gen.generate("expense ratio", [sample_passage])
        # Gracefully falls back to deterministic extraction
        assert "1.47%" in res["answer"]
        assert sample_passage.url in res["answer"]

def test_groq_provider_resolution():
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_test12345"}, clear=True):
        gen = FactsOnlyGenerator()
        assert gen.provider == "groq"
        assert gen.api_key == "gsk_test12345"
        assert gen.model == "llama-3.3-70b-versatile"
        assert gen.base_url == "https://api.groq.com/openai/v1"
        status = gen.get_status()
        assert status["is_active"] is True
        assert "Groq" in status["status_label"]

def test_explicit_provider_selection():
    gen_groq = FactsOnlyGenerator(provider="groq", api_key="gsk_custom_key", model="llama-3.1-8b-instant")
    assert gen_groq.provider == "groq"
    assert gen_groq.api_key == "gsk_custom_key"
    assert gen_groq.model == "llama-3.1-8b-instant"
    assert gen_groq.base_url == "https://api.groq.com/openai/v1"
    assert gen_groq.get_status()["is_active"] is True

    gen_ollama = FactsOnlyGenerator(provider="ollama")
    assert gen_ollama.provider == "ollama"
    assert gen_ollama.base_url == "http://localhost:11434/v1"
    assert gen_ollama.get_status()["is_active"] is True

def test_engine_dynamic_llm_reconfiguration():
    from rag.engine import RAGEngine
    engine = RAGEngine()
    status_initial = engine.get_llm_status()
    assert "provider" in status_initial

    engine.set_llm_config(provider="groq", api_key="gsk_new_key", model="llama-3.3-70b-versatile")
    status_updated = engine.get_llm_status()
    assert status_updated["provider"] == "groq"
    assert status_updated["model"] == "llama-3.3-70b-versatile"
    assert status_updated["is_active"] is True
    assert "Groq" in status_updated["status_label"]

