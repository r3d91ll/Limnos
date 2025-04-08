"""
Graph Elements for GraphRAG

This module defines the GraphNode and GraphEdge classes, which represent the
elements of the NetworkX graph used in the GraphRAG implementation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import uuid4
import networkx as nx

from .entity import Entity
from .relationship import Relationship


@dataclass
class GraphNode:
    """
    Represents a node in the GraphRAG knowledge graph.
    
    Graph nodes are typically derived from entities but may also represent
    other elements such as documents or sections. They contain all the
    information needed for graph operations and retrieval.
    """
    
    # Core attributes
    id: str
    node_type: str
    label: str
    
    # Source information
    source_document_ids: Set[str] = field(default_factory=set)
    
    # Vector representation for semantic search - annotated explicitly to help type checking
    # Use Union to make it clear this can be either None or a list of floats
    embedding: Optional[List[float]] = None
    
    # Additional attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_entity(cls, entity: Entity) -> 'GraphNode':
        """
        Create a GraphNode from an Entity.
        
        Args:
            entity: The entity to convert
            
        Returns:
            A GraphNode representing the entity
        """
        # Ensure we have a non-None label by using an empty string as fallback
        label = entity.canonical_name if entity.canonical_name is not None else ""
        
        node = cls(
            id=entity.id,
            node_type=entity.entity_type,
            label=label
        )
        
        # Add source document if available
        if entity.source_document_id:
            node.source_document_ids.add(entity.source_document_id)
            
        # Add entity attributes
        node.attributes.update({
            "text": entity.text,
            "confidence": entity.confidence,
            "aliases": list(entity.aliases),
            "entity_fingerprint": entity.fingerprint
        })
        
        # Add any additional metadata
        if entity.metadata:
            node.attributes["metadata"] = entity.metadata
            
        return node
    
    def to_networkx_node(self) -> Tuple[str, Dict[str, Any]]:
        """
        Convert to a format suitable for adding to a NetworkX graph.
        
        Returns:
            A tuple of (node_id, node_attributes) for use with nx.add_node()
        """
        attributes = {
            "node_type": self.node_type,
            "label": self.label,
            "source_document_ids": list(self.source_document_ids)
        }
        
        # Add embedding if available
        if self.embedding:
            # First convert to list of strings to satisfy type requirements, then it will be converted back when loaded
            # This is a workaround for type compatibility issues
            attributes["embedding"] = [str(val) for val in self.embedding] if self.embedding else []
            
        # Add all other attributes
        attributes.update(self.attributes)
        
        return self.id, attributes
    
    @classmethod
    def from_networkx_node(cls, node_id: str, attributes: Dict[str, Any]) -> 'GraphNode':
        """
        Create a GraphNode from a NetworkX node.
        
        Args:
            node_id: The node ID in the NetworkX graph
            attributes: The node attributes from the NetworkX graph
            
        Returns:
            A GraphNode instance
        """
        # Extract core attributes
        node_type = attributes.pop("node_type")
        label = attributes.pop("label")
        
        # Extract source documents
        source_document_ids = set(attributes.pop("source_document_ids", []))
        
        # Extract embedding if available and ensure proper typing
        raw_embedding = attributes.pop("embedding", None)
        # Convert embedding to List[float] if it exists
        # Make sure we handle any potential format (strings or numbers)
        embedding: Optional[List[float]] = None
        if raw_embedding is not None:
            try:
                # Convert all values to float to ensure type safety
                embedding = [float(x) for x in raw_embedding]
            except (ValueError, TypeError):
                # If conversion fails, default to None
                embedding = None
        
        # Create the node with remaining attributes
        return cls(
            id=node_id,
            node_type=node_type,
            label=label,
            source_document_ids=source_document_ids,
            embedding=embedding,
            attributes=attributes
        )


@dataclass
class GraphEdge:
    """
    Represents an edge in the GraphRAG knowledge graph.
    
    Graph edges are typically derived from relationships but may also represent
    other connections such as document references or section hierarchies.
    """
    
    # Core attributes
    source_id: str
    target_id: str
    edge_type: str
    
    # Edge properties
    weight: float = 1.0
    bidirectional: bool = False
    
    # Source information
    source_document_ids: Set[str] = field(default_factory=set)
    
    # Additional attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_relationship(cls, relationship: Relationship) -> 'GraphEdge':
        """
        Create a GraphEdge from a Relationship.
        
        Args:
            relationship: The relationship to convert
            
        Returns:
            A GraphEdge representing the relationship
        """
        edge = cls(
            source_id=relationship.source_id,
            target_id=relationship.target_id,
            edge_type=relationship.relationship_type,
            weight=relationship.weight,
            bidirectional=relationship.bidirectional
        )
        
        # Add source document if available
        if relationship.source_document_id:
            edge.source_document_ids.add(relationship.source_document_id)
            
        # Add relationship attributes
        edge.attributes.update({
            "confidence": relationship.confidence,
            "relationship_fingerprint": relationship.fingerprint
        })
        
        # Add context if available
        if relationship.context:
            edge.attributes["context"] = relationship.context
            
        # Add extraction method
        edge.attributes["extraction_method"] = relationship.extraction_method
        
        # Add any additional metadata
        if relationship.metadata:
            edge.attributes["metadata"] = relationship.metadata
            
        return edge
    
    def to_networkx_edge(self) -> Tuple[str, str, Dict[str, Any]]:
        """
        Convert to a format suitable for adding to a NetworkX graph.
        
        Returns:
            A tuple of (source_id, target_id, edge_attributes) for use with nx.add_edge()
        """
        attributes = {
            "edge_type": self.edge_type,
            "weight": self.weight,
            "bidirectional": self.bidirectional,
            "source_document_ids": list(self.source_document_ids)
        }
        
        # Add all other attributes
        attributes.update(self.attributes)
        
        return self.source_id, self.target_id, attributes
    
    @classmethod
    def from_networkx_edge(cls, source_id: str, target_id: str, 
                          attributes: Dict[str, Any]) -> 'GraphEdge':
        """
        Create a GraphEdge from a NetworkX edge.
        
        Args:
            source_id: The source node ID
            target_id: The target node ID
            attributes: The edge attributes from the NetworkX graph
            
        Returns:
            A GraphEdge instance
        """
        # Extract core attributes
        edge_type = attributes.pop("edge_type")
        weight = attributes.pop("weight", 1.0)
        bidirectional = attributes.pop("bidirectional", False)
        
        # Extract source documents
        source_document_ids = set(attributes.pop("source_document_ids", []))
        
        # Create the edge with remaining attributes
        return cls(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            bidirectional=bidirectional,
            source_document_ids=source_document_ids,
            attributes=attributes
        )
