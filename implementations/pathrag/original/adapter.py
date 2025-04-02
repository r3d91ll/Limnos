"""
PathRAG Original Implementation Adapter

This module serves as an adapter for the original PathRAG implementation,
providing a consistent interface for our experimental framework while
using the original codebase. This implementation uses NetworkX for graph
representation and traversal.
"""

import os
import sys
from typing import Dict, List, Any, Optional

# Add the original PathRAG implementation to the Python path
PATHRAG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "temp", "pathrag")
sys.path.append(PATHRAG_DIR)

# Import the original PathRAG implementation
try:
    from PathRAG.PathRAG import PathRAG as OriginalPathRAG
    from PathRAG.utils import init_model, init_tokenizer
    import networkx as nx
except ImportError as e:
    raise ImportError(
        f"Import error: {e}. Original PathRAG implementation or NetworkX not found. "
        f"Make sure the code is available at: {PATHRAG_DIR} and NetworkX is installed."
    )

class PathRAGAdapter:
    """Adapter for the original NetworkX-based PathRAG implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PathRAG adapter with the given configuration.
        
        Args:
            config: Configuration dictionary for PathRAG
        """
        self.config = config
        self.model_name = config.get("model_name", "gpt2")
        self.pathrag = None
        self.graph = None  # NetworkX graph will be initialized by PathRAG
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize the PathRAG system with NetworkX graph."""
        # Initialize the model and tokenizer according to the original implementation
        model = init_model(self.model_name)
        tokenizer = init_tokenizer(self.model_name)
        
        # Initialize the original PathRAG implementation
        self.pathrag = OriginalPathRAG(
            model=model,
            tokenizer=tokenizer,
            **{k: v for k, v in self.config.items() if k not in ["model_name"]}
        )
        
        # The graph is managed internally by PathRAG using NetworkX
        # We can access it for analysis if needed
        self.initialized = True
    
    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Process a query using PathRAG.
        
        Args:
            query: The query string
            **kwargs: Additional arguments for the query
            
        Returns:
            Dict containing the response and related information
        """
        if not self.initialized:
            self.initialize()
            
        # Call the original implementation
        result = self.pathrag.answer(query, **kwargs)
        
        # Return in a standardized format for our framework
        return {
            "answer": result.get("answer", ""),
            "paths": result.get("paths", []),
            "context": result.get("context", ""),
            "raw_result": result  # Include the original result for analysis
        }
    
    def get_paths(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve paths for a query without generating an answer.
        
        Args:
            query: The query string
            top_k: Number of paths to retrieve
            **kwargs: Additional arguments for path retrieval
            
        Returns:
            List of retrieved paths
        """
        if not self.initialized:
            self.initialize()
            
        # Adapt to the original implementation's method for retrieving paths
        paths = self.pathrag.retrieve_paths(query, top_k=top_k, **kwargs)
        
        return paths
    
    def ingest_document(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ingest a document into the knowledge base.
        
        Args:
            document: Document text to ingest
            metadata: Optional metadata for the document
            
        Returns:
            Dict containing information about the ingestion result
        """
        if not self.initialized:
            self.initialize()
            
        # Use the original implementation's method for ingesting documents
        # Note: This will depend on the actual PathRAG API, which might need adaptation
        result = self.pathrag.ingest_text(document, metadata=metadata or {})
        
        return {
            "success": True,
            "nodes_created": result.get("nodes_created", 0),
            "edges_created": result.get("edges_created", 0),
            "document_id": result.get("document_id", ""),
            "raw_result": result
        }
    
    def ingest_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Ingest a file into the knowledge base.
        
        Args:
            file_path: Path to the file to ingest
            **kwargs: Additional arguments for ingestion
            
        Returns:
            Dict containing information about the ingestion result
        """
        if not self.initialized:
            self.initialize()
            
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get metadata from the file path
        metadata = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            **kwargs
        }
        
        # Ingest the document
        return self.ingest_document(content, metadata)
