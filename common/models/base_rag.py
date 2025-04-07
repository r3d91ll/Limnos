"""
Base RAG Interface

This module defines the base interface for all RAG implementations used in the experiments.
All implementations (PathRAG, GraphRAG) across all phases must adhere to this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseRAG(ABC):
    """Base abstract class for all RAG implementations."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the RAG system."""
        pass
    
    @abstractmethod
    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Process a query and generate an answer.
        
        Args:
            query: The query string
            **kwargs: Additional arguments for the query
            
        Returns:
            Dict containing the response and related information
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def ingest_document(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ingest a document into the knowledge base.
        
        Args:
            document: Document text to ingest
            metadata: Optional metadata for the document
            
        Returns:
            Dict containing information about the ingestion result
        """
        pass
    
    @abstractmethod
    def ingest_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Ingest a file into the knowledge base.
        
        Args:
            file_path: Path to the file to ingest
            **kwargs: Additional arguments for ingestion
            
        Returns:
            Dict containing information about the ingestion result
        """
        pass
