try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class EmbeddingModel:
    def __init__(self, model_name: str):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed. Please install it using `pip install sentence-transformers`.")
        
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
    def embed_text(self, text: str):
        """Generates embedding for a single string."""
        return self.model.encode(text)

    def embed_batch(self, texts: list[str]):
        """Generates embeddings for a list of strings."""
        return self.model.encode(texts)
