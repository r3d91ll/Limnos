"""
Neo4j GraphRAG Implementation Adapter

This module serves as an adapter for the Neo4j GraphRAG implementation,
providing a consistent interface for our experimental framework while
using the original codebase.
"""

import os
import sys
from typing import Dict, List, Any, Optional

# Add the original Neo4j GraphRAG implementation to the Python path
GRAPHRAG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "temp", "neo4j-graphrag-python")
sys.path.append(GRAPHRAG_DIR)

# Import the Neo4j GraphRAG implementation
try:
    import neo4j_graphrag
    from neo4j_graphrag.retrievers import SimpleKGPipeline, Pipeline
    from neo4j_graphrag.llm import OpenAILLM
    # We'll be replacing this with Ollama in our Qwen25 implementation
except ImportError:
    raise ImportError(
        "Neo4j GraphRAG implementation not found. "
        "Make sure the code is available at: " + GRAPHRAG_DIR
    )

class GraphRAGAdapter:
    """Adapter for the Neo4j GraphRAG implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GraphRAG adapter with the given configuration.
        
        Args:
            config: Configuration dictionary for GraphRAG
        """
        self.config = config
        self.uri = config.get("neo4j_uri", "bolt://localhost:7687")
        self.username = config.get("neo4j_username", "neo4j")
        self.password = config.get("neo4j_password", "password")
        self.database = config.get("neo4j_database", "neo4j")
        self.openai_api_key = config.get("openai_api_key", "")
        
        self.pipeline = None
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize the GraphRAG system with Neo4j connection."""
        # Configure the Neo4j connection
        try:
            # Create the SimpleKGPipeline (default implementation in Neo4j GraphRAG)
            self.pipeline = SimpleKGPipeline(
                uri=self.uri,
                username=self.username,
                password=self.password,
                database=self.database,
                llm=OpenAILLM(api_key=self.openai_api_key),
                **{k: v for k, v in self.config.items() 
                   if k not in ["neo4j_uri", "neo4j_username", "neo4j_password", 
                                "neo4j_database", "openai_api_key"]}
            )
            self.initialized = True
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Neo4j GraphRAG: {str(e)}")
    
    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Process a query using GraphRAG.
        
        Args:
            query: The query string
            **kwargs: Additional arguments for the query
            
        Returns:
            Dict containing the response and related information
        """
        if not self.initialized:
            self.initialize()
        
        # Get parameters from kwargs with defaults
        top_k = kwargs.get("top_k", 5)
        
        # Execute the query using the pipeline
        result = self.pipeline.execute(
            query=query,
            top_k=top_k,
            **{k: v for k, v in kwargs.items() if k not in ["top_k"]}
        )
        
        # Extract relevant information and return in a standardized format
        return {
            "answer": result.get("answer", ""),
            "context": result.get("context", ""),
            "paths": result.get("paths", []),
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
        
        # Use the retriever directly to get paths without generating an answer
        paths = self.pipeline.retriever.retrieve(
            query=query, 
            top_k=top_k,
            **kwargs
        )
        
        return paths
