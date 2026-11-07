"""
retriever.py
Hybrid text retriever for curated Groww Mutual Fund official sources.
Indexes markdown documents with frontmatter and scores passages using BM25/TF-IDF
with semantic scheme-boosting.
"""

import os
import re
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class Passage:
    doc_id: str
    title: str
    url: str
    organization: str
    category: str
    last_updated: str
    section_title: str
    content: str
    score: float = 0.0

class SourceRetriever:
    def __init__(self, sources_dir: Optional[str] = None):
        if sources_dir is None:
            # Default to project sources/ directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sources_dir = os.path.join(base_dir, "sources")
        self.sources_dir = sources_dir
        self.passages: List[Passage] = []
        self.doc_freq: Dict[str, int] = {}
        self.total_passages: int = 0
        self._load_and_index()

    def _parse_frontmatter(self, text: str) -> Tuple[Dict[str, str], str]:
        meta = {}
        body = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                fm_text = parts[1]
                body = parts[2].strip()
                for line in fm_text.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meta[k.strip()] = v.strip().strip('"').strip("'")
        return meta, body

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r'[^a-zA-Z0-9\s₹%]', ' ', text.lower())
        tokens = [t.strip() for t in cleaned.split() if len(t.strip()) > 1]
        return tokens

    def _load_and_index(self):
        if not os.path.exists(self.sources_dir):
            return

        passages_list = []
        for fname in sorted(os.listdir(self.sources_dir)):
            if not fname.endswith(".md") or fname in ("sources_catalog.md",):
                continue
            fpath = os.path.join(self.sources_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    raw_text = f.read()
            except Exception:
                continue

            meta, body = self._parse_frontmatter(raw_text)
            doc_id = meta.get("id", fname.replace(".md", ""))
            title = meta.get("title", doc_id)
            url = meta.get("url", "")
            org = meta.get("organization", "Groww Mutual Fund")
            cat = meta.get("category", "")
            last_up = meta.get("last_updated_from_sources", "2026-11-07")

            # Split body into sections by markdown headers (## or #)
            sections = re.split(r'\n(?=#{1,3}\s+)', body)
            for sec in sections:
                sec_text = sec.strip()
                if not sec_text or len(sec_text) < 40:
                    continue
                # Extract header line if present
                lines = sec_text.splitlines()
                sec_title = lines[0].lstrip("#").strip() if lines and lines[0].startswith("#") else title
                
                # If section is very long, sub-chunk by double newlines
                if len(sec_text) > 1200:
                    sub_chunks = sec_text.split("\n\n")
                    curr_chunk = []
                    curr_len = 0
                    for sc in sub_chunks:
                        curr_chunk.append(sc)
                        curr_len += len(sc)
                        if curr_len > 800:
                            p_content = "\n\n".join(curr_chunk)
                            passages_list.append(Passage(
                                doc_id=doc_id,
                                title=title,
                                url=url,
                                organization=org,
                                category=cat,
                                last_updated=last_up,
                                section_title=sec_title,
                                content=p_content
                            ))
                            curr_chunk = []
                            curr_len = 0
                    if curr_chunk:
                        p_content = "\n\n".join(curr_chunk)
                        passages_list.append(Passage(
                            doc_id=doc_id,
                            title=title,
                            url=url,
                            organization=org,
                            category=cat,
                            last_updated=last_up,
                            section_title=sec_title,
                            content=p_content
                        ))
                else:
                    passages_list.append(Passage(
                        doc_id=doc_id,
                        title=title,
                        url=url,
                        organization=org,
                        category=cat,
                        last_updated=last_up,
                        section_title=sec_title,
                        content=sec_text
                    ))

        self.passages = passages_list
        self.total_passages = len(passages_list)

        # Build Document Frequency dictionary
        df = {}
        for p in self.passages:
            seen_tokens = set(self._tokenize(p.title + " " + p.section_title + " " + p.content))
            for tok in seen_tokens:
                df[tok] = df.get(tok, 0) + 1
        self.doc_freq = df

    def retrieve(self, query: str, top_k: int = 3) -> List[Passage]:
        if not self.passages:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return self.passages[:top_k]

        q_lower = query.lower()

        # Specific scheme routing boost
        scheme_boosts = {
            "elss": ["elss", "tax saver"],
            "total market": ["total market", "nifty total market"],
            "nifty": ["nifty total market", "nifty 50"],
            "multicap": ["multicap", "multi cap"],
            "liquid": ["liquid fund", "liquid"],
            "banking": ["banking", "financial services"],
            "capital gain": ["capital gains", "tax p&l", "tax report"],
            "statement": ["statement", "cas", "consolidated account statement"],
            "lock-in": ["lock-in", "elss", "3 years", "lock in"],
            "exit load": ["exit load", "redeem"],
            "expense ratio": ["expense ratio", "ter"],
            "riskometer": ["riskometer", "risk-o-meter", "risk level"],
            "benchmark": ["benchmark", "tri", "total return index"],
            "cut-off": ["cut-off", "cut off", "timing", "nav time"],
            "sip": ["start a sip", "cancel my sip", "sip"]
        }

        active_schemes = []
        for key, aliases in scheme_boosts.items():
            if any(alias in q_lower for alias in aliases):
                active_schemes.extend(aliases)

        scored_passages = []
        for p in self.passages:
            p_tokens = self._tokenize(p.title + " " + p.section_title + " " + p.content)
            p_text_lower = (p.title + " " + p.section_title + " " + p.content).lower()
            token_counts = {}
            for t in p_tokens:
                token_counts[t] = token_counts.get(t, 0) + 1

            # BM25-like TF-IDF scoring
            score = 0.0
            doc_len = len(p_tokens)
            k1 = 1.2
            b = 0.75
            avg_doc_len = 120.0

            for q_tok in query_tokens:
                tf = token_counts.get(q_tok, 0)
                if tf > 0:
                    df = self.doc_freq.get(q_tok, 1)
                    idf = math.log((self.total_passages - df + 0.5) / (df + 0.5) + 1.0)
                    tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                    score += idf * tf_norm

            # Scheme / Topic boost
            for alias in active_schemes:
                if alias in p_text_lower:
                    score += 2.5
                if alias in p.title.lower():
                    score += 4.0

            # Direct URL or title matching boost
            if any(qt in p.title.lower() for qt in query_tokens):
                score += 1.5

            if score > 0:
                p_copy = Passage(
                    doc_id=p.doc_id,
                    title=p.title,
                    url=p.url,
                    organization=p.organization,
                    category=p.category,
                    last_updated=p.last_updated,
                    section_title=p.section_title,
                    content=p.content,
                    score=score
                )
                scored_passages.append(p_copy)

        scored_passages.sort(key=lambda x: x.score, reverse=True)
        return scored_passages[:top_k]
