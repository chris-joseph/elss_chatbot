#!/usr/bin/env python3
"""
rebuild_catalog.py
Rebuilds sources_catalog.json, sources_catalog.csv, and sources_catalog.md
from the markdown source files in sources/.
"""

import os
import re
import json
import csv

def parse_frontmatter(text):
    meta = {}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sources_dir = os.path.join(base_dir, "sources")
    
    entries = []
    for fname in sorted(os.listdir(sources_dir)):
        if not fname.endswith(".md") or fname == "sources_catalog.md":
            continue
        fpath = os.path.join(sources_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        meta = parse_frontmatter(content)
        entry = {
            "id": meta.get("id", fname.replace(".md", "")),
            "title": meta.get("title", fname.replace(".md", "")),
            "organization": meta.get("organization", "Groww"),
            "category": meta.get("category", "General"),
            "doc_type": meta.get("doc_type", "concept_guide"),
            "url": meta.get("url", ""),
            "filename": fname
        }
        entries.append(entry)

    # 1. Write sources_catalog.json
    json_path = os.path.join(sources_dir, "sources_catalog.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(entries)} items to {json_path}")

    # 2. Write sources_catalog.csv
    csv_path = os.path.join(sources_dir, "sources_catalog.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "organization", "category", "doc_type", "url", "filename"])
        writer.writeheader()
        writer.writerows(entries)
    print(f"Wrote {len(entries)} items to {csv_path}")

    # 3. Write sources_catalog.md
    md_path = os.path.join(sources_dir, "sources_catalog.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Groww Mutual Fund FAQ Assistant - Official Public Sources Catalog\n\n")
        f.write(f"Total Sources Indexed: **{len(entries)} official public documents**.\n\n")
        f.write("| # | Title | Organization | Category | URL |\n")
        f.write("|---|---|---|---|---|\n")
        for idx, e in enumerate(entries, start=1):
            f.write(f"| {idx} | {e['title']} | {e['organization']} | {e['category']} | [{e['url']}]({e['url']}) |\n")
    print(f"Wrote {len(entries)} items to {md_path}")

if __name__ == "__main__":
    main()
