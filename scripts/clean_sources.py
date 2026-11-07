#!/usr/bin/env python3
"""
clean_sources.py
Cleans noise and trailing footer links from all source documents in sources/*.md.
Retains clean headings, facts, bullet points, procedures, and frontmatter.
"""

import os
import re

SOURCES_DIR = os.path.abspath("sources")

FOOTER_START_MARKERS = [
    "Vaishnavi Tech Park",
    "Share Market Indices",
    "Top Gainers Stocks",
    "Was the answer helpful?",
    "About Us Pricing Blog Media & Press",
    "© 2016- 2026 Groww",
    "© 2016- 2024 Groww",
    "© 2016- 2025 Groww",
    "NIFTY 50 NIFTY Midcap",
    "PRODUCTS Stocks F&O MTF",
    "Calculators IPO Miscellaneous"
]

def clean_source_content(content: str) -> str:
    # Separate frontmatter if present
    frontmatter = ""
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = f"---{parts[1]}---\n\n"
            body = parts[2].strip()

    # Find earliest footer marker in body
    cut_idx = len(body)
    for marker in FOOTER_START_MARKERS:
        idx = body.find(marker)
        if idx != -1 and idx < cut_idx:
            cut_idx = idx

    clean_body = body[:cut_idx].strip()

    # Remove leading web scraper navigation banner
    clean_body = re.sub(
        r'(?s)^\s*(?:[^\n]+\n+)?Invest in stocks, ETFs, IPOs with fast orders.*?(?=\n#|\Z)',
        '',
        clean_body
    ).strip()

    # Remove repeated excessive blank lines
    clean_body = re.sub(r'\n{3,}', '\n\n', clean_body)

    return frontmatter + clean_body + "\n"

def main():
    cleaned_count = 0
    for fname in os.listdir(SOURCES_DIR):
        if not fname.endswith(".md") or fname == "sources_catalog.md":
            continue
        fpath = os.path.join(SOURCES_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            original = f.read()

        cleaned = clean_source_content(original)
        if len(cleaned) < len(original):
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(cleaned)
            cleaned_count += 1
            print(f"[+] Cleaned {fname}: reduced from {len(original)} to {len(cleaned)} chars")

    print(f"\n[DONE] Successfully cleaned {cleaned_count} source documents.")

if __name__ == "__main__":
    main()
