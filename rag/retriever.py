"""
retriever.py
Principled Information Retrieval Engine for Mutual Fund FAQs.
Features:
- Field-weighted TF-IDF vectorization (BM25F field weighting concept).
- Dynamic corpus-guided vocabulary spelling correction (Norvig/Levenshtein style via difflib).
- Query keyword title recall, Jaccard title/section alignment, and verbatim n-gram phrase matching.
- Structural header stub merging to eliminate tiny uninformative chunks.
- Zero hardcoded query-to-answer mappings or query-specific artificial boost dictionaries.
"""

import os
import re
import difflib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter
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

STOPWORDS = {
    "what", "is", "a", "an", "the", "in", "on", "of", "for", "to", "and", "or",
    "how", "do", "i", "can", "my", "me", "are", "by", "with", "at", "from", "its",
    "does", "tell", "about", "which", "there", "any"
}

def normalize_query(query: str, vocabulary: Optional[Set[str]] = None, word_counts: Optional[Counter] = None) -> str:
    """
    Normalizes query text and applies dynamic corpus-guided typo correction.
    If vocabulary is provided, non-stopwords not in vocabulary are matched against
    the indexed corpus vocabulary.
    """
    q_lower = query.lower()
    tokens = re.findall(r'\b\w+\b', q_lower)
    if not vocabulary or not word_counts:
        return " ".join(tokens)

    corrected = []
    for t in tokens:
        if t in vocabulary or len(t) < 4:
            corrected.append(t)
        else:
            matches = difflib.get_close_matches(t, vocabulary, n=3, cutoff=0.75)
            if matches:
                best = max(matches, key=lambda m: word_counts[m])
                corrected.append(best)
            else:
                corrected.append(t)
    return " ".join(corrected)

class SourceRetriever:
    def __init__(self, sources_dir: Optional[str] = None):
        if sources_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sources_dir = os.path.join(base_dir, "sources")
        self.sources_dir = sources_dir
        self.passages: List[Passage] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.word_counts: Counter = Counter()
        self.vocabulary: Set[str] = set()
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

            # Collect corpus vocabulary for typo correction
            for w in re.findall(r'\b[a-zA-Z]{3,}\b', (title + " " + body).lower()):
                self.word_counts[w] += 1

            # Split body into sections by markdown headers
            raw_sections = re.split(r'\n(?=#{1,3}\s+)', body)

            # Merge short header stubs / intros (< 180 chars) into subsequent section
            merged_sections = []
            buffer = ""
            for s in raw_sections:
                s_clean = s.strip()
                if not s_clean:
                    continue
                if len(s_clean) < 180 and not buffer:
                    buffer = s_clean
                    continue
                if buffer:
                    merged_sections.append(buffer + "\n\n" + s_clean)
                    buffer = ""
                else:
                    merged_sections.append(s_clean)
            if buffer:
                merged_sections.append(buffer)

            for sec in merged_sections:
                sec_text = sec.strip()
                if not sec_text or len(sec_text) < 30:
                    continue
                lines = [l.strip() for l in sec_text.splitlines() if l.strip()]
                sec_title = lines[0].lstrip("#").strip() if lines and lines[0].startswith("#") else title

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
        self.vocabulary = set(self.word_counts.keys())
        if not self.passages:
            return

        # Field-Weighted Document Vectorization (BM25F principle):
        # Title (3x) + Section (2x) + Category (1x) + Content (1x)
        corpus_texts = [
            f"{p.title} {p.title} {p.title} {p.section_title} {p.section_title} {p.category} {p.content}"
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
        return {term: int(idx) for term, idx in self.vectorizer.vocabulary_.items()}

    def retrieve(self, query: str, top_k: int = 3) -> List[Passage]:
        if not self.passages or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        # 1. Corpus-Guided Spelling Correction & Keyword Extraction
        q_norm = normalize_query(query, self.vocabulary, self.word_counts)
        q_words = [w for w in re.findall(r'\b\w+\b', q_norm) if len(w) >= 3 and w not in STOPWORDS]
        q_set = set(q_words)
        clean_q_phrase = " ".join(q_words)

        # 2. Vector Space TF-IDF Cosine Similarity
        query_vec = self.vectorizer.transform([q_norm])
        sim_scores = cosine_similarity(self.tfidf_matrix, query_vec).flatten()

        # 3. Principled Field Alignment & Coverage Scoring
        scored_passages = []
        for idx, p in enumerate(self.passages):
            raw_sim = float(sim_scores[idx])

            t_lower = p.title.lower()
            s_lower = p.section_title.lower()
            t_words = set(re.findall(r'\b\w+\b', t_lower))
            t_clean = {w for w in t_words if len(w) >= 3 and w not in STOPWORDS}
            s_words = set(re.findall(r'\b\w+\b', s_lower))
            s_clean = {w for w in s_words if len(w) >= 3 and w not in STOPWORDS}
            c_words = set(re.findall(r'\b\w+\b', p.content.lower()))

            bonus = 0.0
            if q_set:
                # Title keyword recall and Jaccard alignment (topic specificity)
                t_recall = len(q_set & t_clean) / len(q_set)
                t_jaccard = len(q_set & t_clean) / max(1, len(q_set | t_clean))
                s_jaccard = len(q_set & s_clean) / max(1, len(q_set | s_clean))
                c_coverage = len(q_set & c_words) / len(q_set)

                bonus += (0.30 * t_recall) + (0.35 * t_jaccard) + (0.25 * s_jaccard) + (0.20 * c_coverage)

            # Exact N-gram / Phrase Matching
            if clean_q_phrase and (clean_q_phrase in t_lower or clean_q_phrase in s_lower):
                bonus += 0.25

            combined_score = raw_sim + bonus
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
                scored_passages.append(p_copy)

        scored_passages.sort(key=lambda x: x.score, reverse=True)
        return scored_passages[:top_k]
