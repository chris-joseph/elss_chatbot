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
from rag.knowledge_graph import MutualFundKnowledgeGraph
from rag.chroma_store import ChromaDenseStore

class HybridRetriever:
    """
    Tri-modal retrieval engine fusing Knowledge Graph traversal,
    dense vector similarity, and field-weighted BM25F lexical matching.
    """
    def __init__(self, sources_dir: Optional[str] = None):
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
        Executes parallel Tri-Modal retrieval and fuses results via Reciprocal Rank Fusion (RRF).
        Guarantees out-of-scope rejection if lexical confidence is below threshold.
        """
        q_lower = query.lower()
        is_proc = any(w in q_lower for w in ["how to", "how do", "how can", "steps", "cancel", "stop", "download", "start"])

        # 1. Lexical BM25F Stream (with fast OOD gate and exact numerical weighting)
        lex_passages = self.lexical_retriever.retrieve(query, top_k=6, min_score=min_score)
        
        # If the fast OOD gate rejected the query, exit immediately
        if not lex_passages:
            return []

        # 2. Knowledge Graph Stream (Entity extraction & multi-hop edge traversal)
        # Suppress generic concept graph boosts on procedural queries (e.g. how to cancel SIP)
        kg_doc_ids = set()
        if not is_proc:
            kg_res = self.kg.query_graph(query)
            kg_doc_ids = set(kg_res.get("doc_ids", []))

        # 3. Dense Semantic Vector Stream (ChromaDB ONNX embeddings)
        dense_passages = self.dense_store.search(query, top_k=6)

        # 4. Calibrated Reciprocal Rank Fusion (RRF)
        # Formula: RRF_score(d) = sum( w_m / (k + rank_m(d)) )
        k_rrf = 60
        w_lex = 1.2
        w_dense = 0.4
        w_kg = 0.6

        rrf_scores: Dict[str, float] = defaultdict(float)
        passage_map: Dict[str, Passage] = {}

        # Accumulate Lexical Ranks (with confidence-weighted boost)
        for rank, p in enumerate(lex_passages):
            passage_key = f"{p.doc_id}_{p.section_title}"
            passage_map[passage_key] = p
            confidence_factor = 1.0 + max(0.0, float(p.score) - 0.45)
            rrf_scores[passage_key] += (w_lex / (k_rrf + rank + 1)) * confidence_factor

        # Accumulate Dense Ranks
        for rank, p in enumerate(dense_passages):
            passage_key = f"{p.doc_id}_{p.section_title}"
            if passage_key not in passage_map:
                passage_map[passage_key] = p
            rrf_scores[passage_key] += w_dense / (k_rrf + rank + 1)

        # Accumulate Knowledge Graph Multi-Hop Boosts
        if kg_doc_ids:
            for passage_key, p in passage_map.items():
                if p.doc_id in kg_doc_ids:
                    rrf_scores[passage_key] += w_kg / (k_rrf + 1)

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
