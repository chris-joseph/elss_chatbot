"""
retriever.py
Principled Information Retrieval Engine for Mutual Fund FAQs.
Features:
- Field-weighted TF-IDF vectorization (BM25F field weighting concept).
- Morphological lemmatization & plural normalization (indices -> index, stocks -> stock, amcs -> amc).
- Dynamic corpus-guided vocabulary spelling correction (Norvig/Levenshtein style via difflib).
- 2-letter financial acronym support (mf, fd, pe, pb, dp, fy, ay, sd).
- Bidirectional abbreviation synonym expansion (amc <-> asset management company, ter <-> total expense ratio).
- Question intent/relation stopwords filtering (mean, stand, difference, between).
- IDF-weighted keyword title recall, Jaccard alignment, and verbatim n-gram phrase matching.
- Structural header stub merging to eliminate tiny uninformative chunks.
- Fast OOD (out-of-domain) gate and multi-signal relevance threshold gating.
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
    "does", "tell", "about", "which", "there", "any", "between", "into", "through",
    "mean", "means", "meaning", "definition", "define", "stand", "stands",
    "full", "form", "difference", "vs", "versus", "explain", "details",
    "when", "you", "we", "they", "he", "she", "who", "whom", "where", "why"
}

FINANCIAL_SHORT_TERMS: Set[str] = {"mf", "fd", "pe", "pb", "dp", "fy", "ay", "sd"}

PLURAL_MAPPINGS: Dict[str, str] = {
    "indices": "index",
    "indexes": "index",
    "amcs": "amc",
    "sips": "sip",
    "stps": "stp",
    "swps": "swp",
    "funds": "fund",
    "stocks": "stock",
    "folios": "folio",
    "ratios": "ratio",
    "charges": "charge",
    "taxes": "tax",
    "plans": "plan",
    "schemes": "scheme",
    "investments": "investment",
    "timings": "timing",
    "buying": "buy",
    "selling": "sell",
    "transacting": "transact"
}

ABBREVIATION_SYNONYMS: Dict[str, List[str]] = {
    "amc": ["asset", "management", "company"],
    "ter": ["total", "expense", "ratio"],
    "nav": ["net", "asset", "value"],
    "sip": ["systematic", "investment", "plan"],
    "stp": ["systematic", "transfer", "plan"],
    "swp": ["systematic", "withdrawal", "plan"],
    "idcw": ["income", "distribution", "cum", "capital", "withdrawal", "dividend"],
    "elss": ["equity", "linked", "savings", "scheme", "tax", "saver"],
    "cagr": ["compounded", "annual", "growth", "rate"],
    "xirr": ["extended", "internal", "rate", "return"],
    "aum": ["assets", "under", "management"],
    "nfo": ["new", "fund", "offer"],
    "etf": ["exchange", "traded", "fund"],
    "fof": ["fund", "of", "funds"],
    "cas": ["consolidated", "account", "statement"],
    "ecas": ["electronic", "consolidated", "account", "statement"],
    "rta": ["registrar", "transfer", "agent"],
    "kim": ["key", "information", "memorandum"],
    "sid": ["scheme", "information", "document"],
    "sai": ["statement", "additional", "information"],
    "stcg": ["short", "term", "capital", "gains"],
    "ltcg": ["long", "term", "capital", "gains"],
    "stt": ["securities", "transaction", "tax"],
    "tri": ["total", "return", "index"],
    "kyc": ["know", "your", "customer"],
    "ckyc": ["central", "know", "your", "customer"],
    "kra": ["kyc", "registration", "agency"],
    "scores": ["sebi", "complaints", "redress", "system"]
}

MIN_RELEVANCE_SCORE = 0.45

def lemmatize_text(text: str) -> str:
    """Lemmatizes text by mapping known plurals to singular lemmas."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return " ".join(PLURAL_MAPPINGS.get(t, t) for t in tokens)

def normalize_query(query: str, vocabulary: Optional[Set[str]] = None, word_counts: Optional[Counter] = None) -> str:
    """
    Normalizes query text, applies morphological lemmatization,
    and applies dynamic corpus-guided typo correction.
    """
    q_lower = query.lower()
    raw_tokens = re.findall(r'\b\w+\b', q_lower)
    normalized_tokens = [PLURAL_MAPPINGS.get(t, t) for t in raw_tokens]

    if not vocabulary or not word_counts:
        return " ".join(normalized_tokens)

    corrected = []
    for t in normalized_tokens:
        if t in vocabulary or len(t) < 4 or t in FINANCIAL_SHORT_TERMS or t in STOPWORDS:
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
        self.idf_map: Dict[str, float] = {}
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

            # Collect corpus vocabulary with lemmatization
            lemmatized_corpus = lemmatize_text(title + " " + body)
            for w in re.findall(r'\b[a-zA-Z0-9]{2,}\b', lemmatized_corpus):
                if len(w) >= 3 or w in FINANCIAL_SHORT_TERMS:
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

        # Field-Weighted Document Vectorization (BM25F principle) with Lemmatization:
        # Title (3x) + Section (2x) + Category (1x) + Content (1x)
        corpus_texts = []
        for p in self.passages:
            lem_title = lemmatize_text(p.title)
            lem_sec = lemmatize_text(p.section_title)
            lem_cat = lemmatize_text(p.category)
            lem_cont = lemmatize_text(p.content)
            corpus_texts.append(f"{lem_title} {lem_title} {lem_title} {lem_sec} {lem_sec} {lem_cat} {lem_cont}")

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words='english',
            token_pattern=r'(?u)\b\w+\b'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

        # Build fast IDF lookup map
        if hasattr(self.vectorizer, "idf_") and hasattr(self.vectorizer, "vocabulary_"):
            for term, idx in self.vectorizer.vocabulary_.items():
                self.idf_map[term] = float(self.vectorizer.idf_[idx])

    @property
    def total_passages(self) -> int:
        return len(self.passages)

    @property
    def doc_freq(self) -> Dict[str, int]:
        if self.vectorizer is None:
            return {}
        return {term: int(idx) for term, idx in self.vectorizer.vocabulary_.items()}

    def _get_idf(self, term: str) -> float:
        return self.idf_map.get(term, 2.0)

    def _idf_recall(self, q_set: Set[str], target_set: Set[str]) -> float:
        if not q_set:
            return 0.0
        inter = q_set & target_set
        denom = sum(self._get_idf(w) for w in q_set)
        if denom <= 0:
            return 0.0
        numer = sum(self._get_idf(w) for w in inter)
        return numer / denom

    def _idf_jaccard(self, q_set: Set[str], target_set: Set[str]) -> float:
        inter = q_set & target_set
        union = q_set | target_set
        if not union:
            return 0.0
        denom = sum(self._get_idf(w) for w in union)
        if denom <= 0:
            return 0.0
        numer = sum(self._get_idf(w) for w in inter)
        return numer / denom

    def retrieve(self, query: str, top_k: int = 3, min_score: float = MIN_RELEVANCE_SCORE) -> List[Passage]:
        if not self.passages or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        # 1. Corpus-Guided Spelling Correction & Keyword Extraction
        q_norm = normalize_query(query, self.vocabulary, self.word_counts)
        q_words = [
            w for w in re.findall(r'\b\w+\b', q_norm)
            if (len(w) >= 3 or w in FINANCIAL_SHORT_TERMS) and w not in STOPWORDS
        ]
        q_set = set(q_words)
        clean_q_phrase = " ".join(q_words)

        # Fast Out-of-Vocabulary Gate: if query has keywords but NONE exist in corpus vocabulary
        if q_words:
            known_in_corpus = [
                w for w in q_words
                if w in self.vocabulary or any(syn in self.vocabulary for syn in ABBREVIATION_SYNONYMS.get(w, []))
            ]
            if not known_in_corpus:
                return []

        # 2. Enrich query vector with Abbreviation Synonyms (bidirectional)
        expanded_query_tokens = list(q_words)
        for w in q_words:
            if w in ABBREVIATION_SYNONYMS:
                expanded_query_tokens.extend(ABBREVIATION_SYNONYMS[w])
        expanded_query_str = q_norm + " " + " ".join(expanded_query_tokens)

        query_vec = self.vectorizer.transform([expanded_query_str])
        sim_scores = cosine_similarity(self.tfidf_matrix, query_vec).flatten()

        # 3. Principled IDF-Weighted Field Alignment & Coverage Scoring
        scored_passages = []
        for idx, p in enumerate(self.passages):
            raw_sim = float(sim_scores[idx])

            t_lower = lemmatize_text(p.title)
            s_lower = lemmatize_text(p.section_title)
            t_words = set(re.findall(r'\b\w+\b', t_lower))
            t_clean = {w for w in t_words if (len(w) >= 3 or w in FINANCIAL_SHORT_TERMS) and w not in STOPWORDS}
            s_words = set(re.findall(r'\b\w+\b', s_lower))
            s_clean = {w for w in s_words if (len(w) >= 3 or w in FINANCIAL_SHORT_TERMS) and w not in STOPWORDS}
            c_words = set(re.findall(r'\b\w+\b', lemmatize_text(p.content)))

            bonus = 0.0
            if q_set:
                # IDF-weighted Title recall and Jaccard alignment (topic specificity)
                t_recall = self._idf_recall(q_set, t_clean)
                t_jaccard = self._idf_jaccard(q_set, t_clean)
                s_jaccard = self._idf_jaccard(q_set, s_clean)
                c_coverage = self._idf_recall(q_set, c_words)

                bonus += (0.30 * t_recall) + (0.35 * t_jaccard) + (0.25 * s_jaccard) + (0.20 * c_coverage)

            # Exact N-gram / Phrase Matching
            if clean_q_phrase and (clean_q_phrase in t_lower or clean_q_phrase in s_lower):
                bonus += 0.25

            combined_score = raw_sim + bonus
            if combined_score >= min_score:
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
