"""
hybrid_retriever.py
Tri-Modal Hybrid GraphRAG Retriever for Groww Mutual Fund FAQs.
Combines:
1. In-Memory Knowledge Graph (networkx.DiGraph) for deterministic multi-hop relational facts.
2. Local Persistent ChromaDB Dense Vector Store (384-dim ONNX embeddings) for semantic paraphrasing.
3. Field-Weighted Lexical BM25F Engine (scipy.sparse.csr_matrix) for exact numerical precision and out-of-scope gating.
Fuses all signals using Reciprocal Rank Fusion (RRF).
"""

import os
from typing import List, Dict, Optional, Any, Set
from collections import defaultdict
from rag.retriever import SourceRetriever, Passage, MIN_RELEVANCE_SCORE

try:
    from rag.knowledge_graph import MutualFundKnowledgeGraph
except (ImportError, ModuleNotFoundError):
    MutualFundKnowledgeGraph = None

try:
    from rag.chroma_store import ChromaDenseStore
except (ImportError, ModuleNotFoundError):
    ChromaDenseStore = None

class HybridRetriever:
    """
    Tri-modal retrieval engine fusing Knowledge Graph traversal,
    dense vector similarity, and field-weighted BM25F lexical matching.
    """
    def __init__(self, sources_dir: Optional[str] = None):
        if MutualFundKnowledgeGraph is None or ChromaDenseStore is None:
            raise ImportError(
                "Missing required dependencies (networkx or chromadb) for HybridRetriever. "
                "Install with: pip install networkx chromadb"
            )
        self.lexical_retriever = SourceRetriever(sources_dir=sources_dir)
        self.kg = MutualFundKnowledgeGraph(sources_dir=sources_dir)
        self.dense_store = ChromaDenseStore()
        
        # Ensure ChromaDB index is warm and in sync with passages
        if self.dense_store.count() != len(self.lexical_retriever.passages):
            self.dense_store.index_passages(self.lexical_retriever.passages)

    @property
    def passages(self) -> List[Passage]:
        return self.lexical_retriever.passages

    @property
    def total_passages(self) -> int:
        return len(self.passages)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = MIN_RELEVANCE_SCORE
    ) -> List[Passage]:
        """
        Streamlined Vector-First Hybrid Retriever.
        Fuses dense semantic similarity (ChromaDB) with lexical keyword matching (BM25)
        and Knowledge Graph multi-hop candidate injection.
        """
        q_lower = query.lower()
        is_proc = any(w in q_lower for w in ["how to", "how do", "how can", "steps", "cancel", "stop", "download", "start"])

        # Fast Out-of-Vocabulary Gate: abort early if all non-stopwords are out-of-corpus
        if self.lexical_retriever.is_out_of_vocabulary(query):
            return []

        # 1. Dense Semantic Vector Stream (ChromaDB ONNX embeddings) - Primary
        dense_passages = self.dense_store.search(query, top_k=8)

        # 2. Lexical BM25F Stream (with exact numerical weighting) - Secondary
        lex_passages = self.lexical_retriever.retrieve(query, top_k=8, min_score=min_score)

        # Dual-Stream Out-of-Scope Gate: if lexical found nothing AND dense similarity < 0.30
        if not lex_passages and (not dense_passages or dense_passages[0].score < 0.30):
            return []

        # 3. Knowledge Graph Stream (Entity extraction & multi-hop edge traversal)
        kg_doc_ids = set()
        if not is_proc:
            kg_res = self.kg.query_graph(query)
            kg_doc_ids = set(kg_res.get("doc_ids", []))

        # 4. Calibrated Reciprocal Rank Fusion (RRF) with KG Candidate Injection
        k_rrf = 60
        w_lex = 1.2
        w_dense = 1.0
        w_kg = 0.8

        rrf_scores: Dict[str, float] = defaultdict(float)
        passage_map: Dict[str, Passage] = {}

        # Accumulate Dense Ranks (primary semantic signal)
        for rank, p in enumerate(dense_passages):
            passage_key = f"{p.doc_id}_{p.section_title}"
            passage_map[passage_key] = p
            rrf_scores[passage_key] += w_dense / (k_rrf + rank + 1)

        # Accumulate Lexical Ranks (with confidence-weighted boost)
        for rank, p in enumerate(lex_passages):
            passage_key = f"{p.doc_id}_{p.section_title}"
            if passage_key not in passage_map:
                passage_map[passage_key] = p
            confidence_factor = 1.0 + max(0.0, float(p.score) - 0.45)
            rrf_scores[passage_key] += (w_lex / (k_rrf + rank + 1)) * confidence_factor

        # Accumulate Knowledge Graph Multi-Hop Boosts & Candidate Injection
        if kg_doc_ids:
            # Scheme and AMC entities get primary priority over secondary relational attribute nodes
            primary_entity_doc_ids = {
                self.kg.graph.nodes[n].get("doc_id")
                for n in kg_res.get("matched_entities", [])
                if self.kg.graph.nodes[n].get("type") in ("Scheme", "AMC")
            }

            # First, boost existing passages in passage_map matching kg_doc_ids
            for passage_key, p in passage_map.items():
                if p.doc_id in primary_entity_doc_ids:
                    rrf_scores[passage_key] += (w_kg * 1.35) / (k_rrf + 1)
                elif p.doc_id in kg_doc_ids:
                    rrf_scores[passage_key] += w_kg / (k_rrf + 1)

            # Candidate Injection: If a kg_doc_id is not yet in passage_map, inject its top passage
            for p in self.lexical_retriever.passages:
                if p.doc_id in kg_doc_ids:
                    passage_key = f"{p.doc_id}_{p.section_title}"
                    if passage_key not in passage_map:
                        passage_map[passage_key] = p
                        boost = (w_kg * 1.35) if p.doc_id in primary_entity_doc_ids else w_kg
                        rrf_scores[passage_key] += boost / (k_rrf + 1)

        # Sort candidate passages by fused RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        fused_passages: List[Passage] = []
        for key in sorted_keys[:top_k]:
            p = passage_map[key]
            # Attach normalized fusion score
            p_copy = Passage(
                doc_id=p.doc_id,
                title=p.title,
                url=p.url,
                organization=p.organization,
                category=p.category,
                last_updated=p.last_updated,
                section_title=p.section_title,
                content=p.content,
                score=round(rrf_scores[key], 4)
            )
            fused_passages.append(p_copy)

        return fused_passages
