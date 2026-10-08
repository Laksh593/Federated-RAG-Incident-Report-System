import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

class HybridVectorStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize persistent ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(self.directory))
        self.collection = self.chroma_client.get_or_create_collection(
            name="incident_policies",
            metadata={"hnsw:space": "cosine"}
        )
        
        # BM25 is kept in-memory and re-populated/loaded when documents change
        self.bm25 = None
        self.documents = []  # To keep track of raw docs for BM25
        
    def add_documents(self, embeddings: List[List[float]], docs: List[Dict[str, Any]]):
        """
        Add embeddings and corresponding documents to ChromaDB and BM25 idempotently.
        """
        if not docs:
            return

        # 1. Add to ChromaDB via upsert (idempotent, avoids duplicates / constraint errors)
        ids = [f"doc_{doc['source']}_{doc.get('chunk_id', i)}" for i, doc in enumerate(docs)]
        texts = [doc['content'] for doc in docs]
        metadatas = [{"source": doc['source'], "chunk_id": doc.get('chunk_id', 0)} for doc in docs]
        
        self.collection.upsert(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # 2. Add/update in BM25 with key-based deduplication
        doc_map = {}
        for idx, d in enumerate(self.documents):
            key = (d.get('source'), d.get('chunk_id', idx))
            doc_map[key] = d
        for i, doc in enumerate(docs):
            key = (doc.get('source'), doc.get('chunk_id', i))
            doc_map[key] = doc

        self.documents = list(doc_map.values())
        tokenized_corpus = [doc['content'].lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def search(self, query: str, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs Hybrid Search (Dense ChromaDB + Sparse BM25) and combines via Reciprocal Rank Fusion (RRF).
        """
        # Safety check if DB is empty
        if self.collection.count() == 0:
            return []
            
        # 1. Vector Search (Dense)
        chroma_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        dense_hits = []
        if chroma_results['ids'] and len(chroma_results['ids'][0]) > 0:
            for i in range(len(chroma_results['ids'][0])):
                dense_hits.append({
                    "id": chroma_results['ids'][0][i],
                    "content": chroma_results['documents'][0][i],
                    "metadata": chroma_results['metadatas'][0][i],
                    "dense_distance": chroma_results['distances'][0][i]  # Cosine distance (lower is better)
                })
                
        # 2. BM25 Search (Sparse)
        sparse_hits = []
        if self.bm25 is not None:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Get top k indices
            import numpy as np
            top_indices = np.argsort(bm25_scores)[::-1][:top_k]
            
            for idx in top_indices:
                score = float(bm25_scores[idx])
                if score > 0:  # Only count if there's actually a keyword match
                    doc = self.documents[idx]
                    doc_id = f"doc_{doc['source']}_{doc.get('chunk_id', idx)}"
                    sparse_hits.append({
                        "id": doc_id,
                        "content": doc['content'],
                        "metadata": {"source": doc['source'], "chunk_id": doc.get('chunk_id', idx)},
                        "sparse_score": score # BM25 score (higher is better)
                    })
                    
        # 3. Reciprocal Rank Fusion (RRF)
        # RRF Score = 1 / (k + rank)
        k_rf = 60
        rrf_scores = {}
        unified_docs = {}
        
        # Rank Dense hits (remember chroma returns distance, so we already sorted it ascending)
        for rank, hit in enumerate(dense_hits):
            doc_id = hit["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1.0 / (k_rf + rank + 1)
            unified_docs[doc_id] = hit
            
        # Rank Sparse hits (sorted descending by score)
        for rank, hit in enumerate(sparse_hits):
            doc_id = hit["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1.0 / (k_rf + rank + 1)
            if doc_id not in unified_docs:
                unified_docs[doc_id] = hit
                
        # Sort by final RRF score
        sorted_rrf = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Format output to match old API
        final_results = []
        for doc_id, score in sorted_rrf[:top_k]:
            doc_info = unified_docs[doc_id]
            final_results.append({
                "document": {
                    "content": doc_info["content"],
                    "source": doc_info["metadata"]["source"]
                },
                "score": score,
                "type": "hybrid_rrf"
            })
            
        return final_results

    def save(self):
        """ChromaDB auto-saves. Just saving BM25 state."""
        with open(self.directory / "bm25_state.pkl", "wb") as f:
            pickle.dump(self.documents, f)
            
    def load(self):
        """ChromaDB auto-loads from directory. Just loading BM25 state."""
        bm25_path = self.directory / "bm25_state.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                raw_docs = pickle.load(f)
                if raw_docs:
                    seen = set()
                    deduped = []
                    for idx, doc in enumerate(raw_docs):
                        key = (doc.get('source'), doc.get('chunk_id', idx))
                        if key not in seen:
                            seen.add(key)
                            deduped.append(doc)
                    self.documents = deduped
                    tokenized_corpus = [doc['content'].lower().split() for doc in self.documents]
                    self.bm25 = BM25Okapi(tokenized_corpus)
            return True
        return False
