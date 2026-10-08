from pathlib import Path
from typing import List, Dict
from .config import config
from .embeddings import EmbeddingModel
from .vector_store import HybridVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

class Retriever:
    def __init__(self):
        self.embedding_model = EmbeddingModel(config.embedding_model_name)
        self.vector_store = HybridVectorStore(config.vector_store_path)
        
        # Load a high-accuracy Cross-Encoder for re-ranking
        if CrossEncoder is not None:
            print("Loading Cross-Encoder re-ranker (ms-marco-MiniLM-L-6-v2)...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        else:
            self.cross_encoder = None
            print("WARNING: sentence-transformers not installed. Re-ranking disabled.")
        
        # Try to load existing index
        if not self.vector_store.load():
            print("No existing vector store found. Please index documents first.")
            
    def index_documents(self, documents: List[Dict[str, str]]):
        """
        documents: List of dicts with 'content' and optional metadata
        """
        texts = [d['content'] for d in documents]
        # Embed in batches (a real implementation would batch this properly if > 1000 docs)
        embeddings = self.embedding_model.embed_batch(texts)
        if hasattr(embeddings, 'tolist'):
            embeddings = embeddings.tolist()
        
        self.vector_store.add_documents(embeddings, documents)
        self.vector_store.save()
        print(f"Indexed {len(documents)} chunks via Hybrid Search.")
        
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        if top_k is None:
            top_k = config.top_k
            
        fetch_k = top_k * 3 # Overfetch for re-ranking (Top 10-15)
            
        query_embedding = self.embedding_model.embed_text(query)
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
            
        # Search via hybrid RRF
        results = self.vector_store.search(query, query_embedding, top_k=fetch_k)
        
        # Re-ranker Phase
        if self.cross_encoder is not None and len(results) > 0:
            print(f"Re-ranking top {len(results)} candidates using Cross-Encoder...")
            # Prepare pairs of (Query, Document)
            sentence_combinations = [[query, res['document']['content']] for res in results]
            scores = self.cross_encoder.predict(sentence_combinations)
            
            # Update scores and sort
            for i, score in enumerate(scores):
                results[i]['score'] = float(score)
                results[i]['type'] = "cross_encoder_reranked"
                
            # Sort descending by cross encoder score
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
        # Return final top_k
        return results[:top_k]

    def ingest_from_directory(self, directory: Path):
        """Ingests .txt files and uses Semantic Overlapping Chunking"""
        docs = []
        if not directory.exists():
            print(f"Directory {directory} does not exist.")
            return

        print(f"Chunking files in {directory} securely using recursive text splitting...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )

        for file_path in directory.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
                # Intelligent overlapping chunks
                chunks = text_splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        docs.append({
                            "content": chunk.strip(),
                            "source": file_path.name,
                            "chunk_id": i
                        })
        
        if docs:
            self.index_documents(docs)
