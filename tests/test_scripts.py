"""
Unit tests for data cleaning and download processing scripts.
"""

import pytest
from scripts.clean_sources import clean_source_content
from scripts.download_sources import clean_html_to_text, format_scheme_data

def test_clean_source_content():
    sample_text = (
        "---\nid: sample_doc\n---\n\n"
        "# Main Heading\n\n"
        "This is valid article body text.\n\n"
        "Was the answer helpful? Vaishnavi Tech Park, South Tower, Bellandur\n"
        "Top Gainers Stocks Share Market Indices"
    )
    cleaned = clean_source_content(sample_text)
    assert "# Main Heading" in cleaned
    assert "This is valid article body text." in cleaned
    assert "Vaishnavi Tech Park" not in cleaned
    assert "Top Gainers" not in cleaned
    assert "---\nid: sample_doc\n---" in cleaned

def test_clean_html_to_text():
    raw_html = (
        "<html><head><script>var x = 1;</script><style>body { color: red; }</style></head>"
        "<body>"
        "<h1>Scheme Overview</h1>"
        "<p>This is a paragraph &amp; more details.</p>"
        "<ul><li>Point 1</li><li>Point 2</li></ul>"
        "</body></html>"
    )
    text = clean_html_to_text(raw_html)
    assert "# Scheme Overview" in text
    assert "This is a paragraph & more details." in text
    assert "* Point 1" in text
    assert "var x = 1" not in text
    assert "color: red" not in text

def test_format_scheme_data():
    sample_next_data = {
        "props": {
            "pageProps": {
                "mfServerSideData": {
                    "scheme_name": "Groww ELSS Tax Saver Fund Direct Growth",
                    "fund_house": "Groww Mutual Fund",
                    "category": "Equity",
                    "sub_category": "ELSS",
                    "expense_ratio": "1.47",
                    "exit_load": "Nil",
                    "lock_in": {"years": 3, "months": 0, "days": 0},
                    "min_sip_investment": 500,
                    "min_investment_amount": 500,
                    "benchmark_name": "NIFTY 500 Total Return Index",
                    "stamp_duty": "0.005%",
                    "description": "Tax saving equity mutual fund scheme.",
                    "fund_manager": "Paras Matalia",
                    "launch_date": "2020-01-01"
                }
            }
        }
    }
    meta = {
        "url": "https://groww.in/mutual-funds/groww-elss-tax-saver-fund-direct-growth",
        "scheme_name": "Groww ELSS Tax Saver Fund Direct Growth"
    }
    
    formatted_md = format_scheme_data(sample_next_data, meta)
    assert formatted_md is not None
    assert "Groww ELSS Tax Saver Fund Direct Growth" in formatted_md
    assert "1.47%" in formatted_md
    assert "Nil" in formatted_md
    assert "3 Years" in formatted_md
    assert "₹500" in formatted_md
    assert "NIFTY 500 Total Return Index" in formatted_md
    assert meta["url"] in formatted_md
