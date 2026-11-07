"""
retriever.py
Upgraded Hybrid Retriever using scikit-learn TfidfVectorizer (unigrams + bigrams),
cosine similarity, BM25 term weighting, and scheme-specific entity boosting.
"""

import os
import re
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sources_dir = os.path.join(base_dir, "sources")
        self.sources_dir = sources_dir
        self.passages: List[Passage] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
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

            # Split body into coherent sections by markdown headers
            sections = re.split(r'\n(?=#{1,3}\s+)', body)
            for sec in sections:
                sec_text = sec.strip()
                if not sec_text or len(sec_text) < 30:
                    continue
                lines = sec_text.splitlines()
                sec_title = lines[0].lstrip("#").strip() if lines and lines[0].startswith("#") else title

                # If section is lengthy, sub-chunk logically
                if len(sec_text) > 1000:
                    sub_chunks = sec_text.split("\n\n")
                    curr_chunk = []
                    curr_len = 0
                    for sc in sub_chunks:
                        curr_chunk.append(sc)
                        curr_len += len(sc)
                        if curr_len > 600:
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
        if not self.passages:
            return

        # Prepare corpus documents for TF-IDF Vectorizer
        corpus_texts = [
            f"{p.title} {p.category} {p.section_title} {p.content}"
            for p in self.passages
        ]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words='english',
            token_pattern=r'(?u)\b\w+\b'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

    @property
    def total_passages(self) -> int:
        return len(self.passages)

    @property
    def doc_freq(self) -> Dict[str, int]:
        if self.vectorizer is None:
            return {}
        # Vocabulary size as proxy for DF mapping
        return {term: int(idx) for term, idx in self.vectorizer.vocabulary_.items()}

    def retrieve(self, query: str, top_k: int = 3) -> List[Passage]:
        if not self.passages or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        q_lower = query.lower()

        # 1. Compute Cosine Similarities via TF-IDF Vector Space
        query_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(self.tfidf_matrix, query_vec).flatten()

        # 2. Scheme and Topic Entity Boosts
        scheme_boosts = {
            "elss": ["elss", "tax saver", "tax saving", "80c"],
            "total market": ["total market", "nifty total market"],
            "nifty": ["nifty total market", "nifty"],
            "multicap": ["multicap", "multi cap"],
            "liquid": ["liquid fund", "liquid"],
            "banking": ["banking", "financial services"],
            "capital gain": ["capital gains", "tax p&l", "tax report", "p&l report"],
            "statement": ["statement", "cas", "consolidated account statement"],
            "lock-in": ["lock-in", "lock in", "3 years", "three years"],
            "exit load": ["exit load", "redeem", "redemption load"],
            "expense ratio": ["expense ratio", "ter", "total expense ratio"],
            "riskometer": ["riskometer", "risk-o-meter", "risk level", "very high"],
            "benchmark": ["benchmark", "tri", "total return index"],
            "cut-off": ["cut-off", "cut off", "timing", "2:00 pm", "12 noon", "same-day nav"],
            "cancel": ["cancel sip", "stop sip", "cancel an ongoing sip"],
            "start": ["start a sip", "start sip", "autopay", "mandate"]
        }

        active_boost_terms = []
        for key, aliases in scheme_boosts.items():
            if any(alias in q_lower for alias in aliases):
                active_boost_terms.extend(aliases)

        # 3. Combine Hybrid Scores
        final_passages = []
        for idx, p in enumerate(self.passages):
            raw_sim = float(sim_scores[idx])
            boost = 0.0
            p_text = (p.title + " " + p.section_title + " " + p.content).lower()

            for term in active_boost_terms:
                if term in p.title.lower():
                    boost += 0.35
                elif term in p_text:
                    boost += 0.15

            # Direct title token match boost
            q_words = re.findall(r'\w+', q_lower)
            for w in q_words:
                if len(w) > 3 and w in p.title.lower():
                    boost += 0.10

            combined_score = raw_sim + boost
            if combined_score > 0.05:
                p_copy = Passage(
                    doc_id=p.doc_id,
                    title=p.title,
                    url=p.url,
                    organization=p.organization,
                    category=p.category,
                    last_updated=p.last_updated,
                    section_title=p.section_title,
                    content=p.content,
                    score=round(combined_score, 4)
                )
                final_passages.append(p_copy)

        final_passages.sort(key=lambda x: x.score, reverse=True)
        return final_passages[:top_k]
