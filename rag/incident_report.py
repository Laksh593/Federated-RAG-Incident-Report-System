from .retriever import Retriever
from .generator import Generator
from .config import config

class IncidentReportGenerator:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        
    def initialize_knowledge_base(self):
        """Re-scan and ingest documents from the configured knowledge base directory."""
        print(f"Ingesting knowledge base from: {config.knowledge_base_dir}")
        self.retriever.ingest_from_directory(config.knowledge_base_dir)
        
    def create_report(self, anomaly_data: dict) -> str:
        """
        Full RAG pipeline: 
        1. Formulate query from anomaly data
        2. Retrieve context
        3. Generate report
        """
        query = f"{anomaly_data.get('type')} {anomaly_data.get('description', '')} handling procedure security policy"
        print(f"Searching knowledge base for: '{query}'")
        
        context_docs = self.retriever.retrieve(query)
        
        if not context_docs:
            print("Warning: No relevant context found in knowledge base.")
            
        report = self.generator.generate_report(anomaly_data, context_docs)
        return report

    def create_report_from_video(self, anomaly_data: dict, video_path: str) -> str:
        """
        Full RAG pipeline with multimodal video input: 
        1. Formulate query from anomaly data
        2. Retrieve context
        3. Generate report analyzing video evidence
        """
        query = f"{anomaly_data.get('type')} {anomaly_data.get('description', '')} handling procedure security policy"
        print(f"Searching knowledge base for multi-modal video context: '{query}'")
        
        context_docs = self.retriever.retrieve(query)
        
        if not context_docs:
            print("Warning: No relevant text context found in knowledge base.")
            
        report = self.generator.generate_report_from_video(video_path, anomaly_data, context_docs)
        return report
