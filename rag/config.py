import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class RAGConfig:
    # Paths
    project_root: Path = Path(__file__).parent.parent
    knowledge_base_dir: Path = project_root / "data" / "knowledge_base"
    vector_store_path: Path = project_root / "data" / "vector_store"
    
    # Embeddings
    embedding_model_name: str = "all-MiniLM-L6-v2"  # Fast and efficient for local use
    embedding_dim: int = 384
    
    # Retrieval
    top_k: int = 3
    similarity_threshold: float = 0.5
    
    # Generation (Local LLM via Ollama)
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_text_model: str = os.environ.get("OLLAMA_TEXT_MODEL", "llama3")    # For text-only incident reports
    llm_vision_model: str = os.environ.get("OLLAMA_VISION_MODEL", "llava")   # For multimodal video analysis
    
    def __post_init__(self):
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)

config = RAGConfig()
