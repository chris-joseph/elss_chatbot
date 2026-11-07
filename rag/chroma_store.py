"""
chroma_store.py
Persistent local embedded ChromaDB vector store for Groww Mutual Fund RAG.
Uses ChromaDB's native ONNX-based local embedding function (all-MiniLM-L6-v2, 384 dimensions).
Provides dense semantic vector similarity matching with zero cloud dependencies.
"""

import os
from typing import List, Dict, Optional, Any

try:
    import chromadb
    from chromadb.utils import embedding_functions
except (ImportError, ModuleNotFoundError):
    chromadb = None
    embedding_functions = None

from rag.retriever import Passage

class ChromaDenseStore:
    """
    Manages persistent local ChromaDB collection for dense semantic search.
    """
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "groww_mutual_funds"
    ):
        if chromadb is None or embedding_functions is None:
            raise ImportError(
                "chromadb is required for ChromaDenseStore. "
                "Install with: pip install chromadb"
            )
        if persist_directory is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            persist_directory = os.path.join(base_dir, "vector_store", "chroma")
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        """Returns the number of indexed passages in ChromaDB."""
        return self.collection.count()

    def index_passages(self, passages: List[Passage], force_reindex: bool = False):
        """
        Indexes passages into ChromaDB. Skips indexing if collection is already populated
        unless force_reindex is True.
        """
        if not passages:
            return

        current_count = self.collection.count()
        if current_count == len(passages) and not force_reindex:
            return

        # If reindexing or counts mismatch, reset collection
        if force_reindex and current_count > 0:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )

        ids = []
        documents = []
        metadatas = []

        for idx, p in enumerate(passages):
            chunk_id = f"{p.doc_id}_{idx}"
            # Enrich text with title and section for optimal semantic dense matching
            doc_text = f"Title: {p.title}\nSection: {p.section_title}\n\n{p.content}"
            ids.append(chunk_id)
            documents.append(doc_text)
            metadatas.append({
                "doc_id": p.doc_id,
                "title": p.title,
                "url": p.url,
                "organization": p.organization,
                "category": p.category,
                "last_updated": p.last_updated,
                "section_title": p.section_title,
                "passage_index": idx
            })

        # Batch upsert in chunks of 100 to avoid payload limits
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self.collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end]
            )

    def search(self, query: str, top_k: int = 5) -> List[Passage]:
        """
        Performs dense semantic similarity search against indexed passages.
        Returns Passages scored by cosine similarity (1.0 - cosine_distance).
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )

        matched_passages: List[Passage] = []
        if not results or not results["documents"] or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else []
        dists = results["distances"][0] if results["distances"] else [0.5] * len(docs)

        for doc_text, meta, dist in zip(docs, metas, dists):
            # Distance in cosine space is between 0 (identical) and 2 (opposite)
            sim_score = max(0.0, 1.0 - float(dist))
            
            # Extract content from doc_text
            content = doc_text
            if "\n\n" in doc_text:
                content = doc_text.split("\n\n", 1)[1]

            p = Passage(
                doc_id=meta.get("doc_id", ""),
                title=meta.get("title", ""),
                url=meta.get("url", ""),
                organization=meta.get("organization", "Groww Mutual Fund"),
                category=meta.get("category", ""),
                last_updated=meta.get("last_updated", "2026-11-07"),
                section_title=meta.get("section_title", ""),
                content=content,
                score=round(sim_score, 4)
            )
            matched_passages.append(p)

        return matched_passages
