"""
Query Models for GraphRAG

This module defines the Query and QueryResult classes, which represent
queries and their results in the GraphRAG implementation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from uuid import uuid4
from datetime import datetime
import networkx as nx

from .entity import Entity
from .relationship import Relationship


@dataclass
class Query:
    """
    Represents a query in the GraphRAG system.
    
    Queries are processed to extract entities and relationships, which are then
    used to search the graph for relevant information.
    """
    
    # Core attributes
    query_text: str
    query_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Extracted components
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    
    # Query parameters
    max_results: int = 10
    max_hops: int = 2
    min_confidence: float = 0.5
    
    # Query context
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Query metadata
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the query to a dictionary representation.
        
        Returns:
            Dictionary representation of the query
        """
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "entities": [entity.to_dict() for entity in self.entities],
            "relationships": [rel.to_dict() for rel in self.relationships],
            "max_results": self.max_results,
            "max_hops": self.max_hops,
            "min_confidence": self.min_confidence,
            "context": self.context,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Query':
        """
        Create a Query instance from a dictionary.
        
        Args:
            data: Dictionary representation of a query
            
        Returns:
            Query instance
        """
        # Create a copy to avoid modifying the input
        query_data = data.copy()
        
        # Handle entities and relationships
        entities = [Entity.from_dict(entity_data) 
                   for entity_data in query_data.pop("entities", [])]
        relationships = [Relationship.from_dict(rel_data) 
                        for rel_data in query_data.pop("relationships", [])]
        
        # Handle created_at
        created_at_str = query_data.pop("created_at", None)
        if created_at_str and isinstance(created_at_str, str):
            try:
                created_at = datetime.fromisoformat(created_at_str)
            except ValueError:
                created_at = datetime.now()
        else:
            created_at = datetime.now()
        
        # Create the query
        query = cls(**query_data, created_at=created_at)
        query.entities = entities
        query.relationships = relationships
        
        return query


@dataclass
class QueryResult:
    """
    Represents the result of a query in the GraphRAG system.
    
    QueryResults contain the retrieved subgraphs, ranked documents, and
    other information needed to generate a response.
    """
    
    # Core attributes
    query_id: str
    
    # Result data
    subgraphs: List[nx.Graph] = field(default_factory=list)
    ranked_documents: List[Dict[str, Any]] = field(default_factory=list)
    
    # Result metadata
    execution_time_ms: float = 0.0
    total_nodes_retrieved: int = 0
    total_edges_retrieved: int = 0
    
    # Additional information
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_subgraph(self, subgraph: nx.Graph) -> None:
        """
        Add a subgraph to the results.
        
        Args:
            subgraph: NetworkX graph representing a retrieved subgraph
        """
        self.subgraphs.append(subgraph)
        self.total_nodes_retrieved += subgraph.number_of_nodes()
        self.total_edges_retrieved += subgraph.number_of_edges()
    
    def add_ranked_document(self, document_id: str, score: float, 
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a ranked document to the results.
        
        Args:
            document_id: ID of the document
            score: Relevance score
            metadata: Optional additional metadata
        """
        doc_entry = {
            "document_id": document_id,
            "score": score
        }
        
        if metadata:
            doc_entry["metadata"] = metadata
            
        self.ranked_documents.append(doc_entry)
        
    def get_top_documents(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the top-ranked documents.
        
        Args:
            limit: Maximum number of documents to return (None for all)
            
        Returns:
            List of top-ranked documents
        """
        sorted_docs = sorted(self.ranked_documents, 
                            key=lambda x: x["score"], reverse=True)
        
        if limit is not None:
            return sorted_docs[:limit]
        return sorted_docs
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the query result to a dictionary representation.
        
        Note: This does not include the subgraphs, as NetworkX graphs are not
        directly serializable. Use a dedicated serialization method for subgraphs.
        
        Returns:
            Dictionary representation of the query result
        """
        return {
            "query_id": self.query_id,
            "ranked_documents": self.ranked_documents,
            "execution_time_ms": self.execution_time_ms,
            "total_nodes_retrieved": self.total_nodes_retrieved,
            "total_edges_retrieved": self.total_edges_retrieved,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryResult':
        """
        Create a QueryResult instance from a dictionary.
        
        Note: This does not reconstruct subgraphs. Subgraphs must be added
        separately using the add_subgraph method.
        
        Args:
            data: Dictionary representation of a query result
            
        Returns:
            QueryResult instance
        """
        # Create a copy to avoid modifying the input
        result_data = data.copy()
        
        # Create the query result
        return cls(**result_data)
