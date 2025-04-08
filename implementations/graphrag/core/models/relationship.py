"""
Relationship Model for GraphRAG

This module defines the Relationship class, which represents connections
between entities in the GraphRAG implementation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from uuid import uuid4
import hashlib
import json


@dataclass
class Relationship:
    """
    Represents a relationship between two entities in the GraphRAG knowledge graph.
    
    Relationships are the edges in the graph, connecting entity nodes and defining
    how they relate to each other. Relationships can be explicit (directly stated in text)
    or implicit (inferred from context or co-occurrence).
    """
    
    # Core attributes
    source_id: str
    target_id: str
    relationship_type: str
    
    # Optional attributes with defaults
    id: str = field(default_factory=lambda: str(uuid4()))
    weight: float = 1.0
    confidence: float = 1.0
    bidirectional: bool = False
    source_document_id: Optional[str] = None
    
    # Contextual information
    context: Optional[str] = None
    
    # Positional information (where this relationship was found in text)
    positions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Provenance information
    extraction_method: str = "explicit"  # explicit, implicit, inferred
    
    @property
    def fingerprint(self) -> str:
        """
        Generate a unique fingerprint for this relationship based on its core attributes.
        
        This is useful for deduplication and identification across documents.
        """
        # For bidirectional relationships, we want the same fingerprint regardless of direction
        if self.bidirectional and self.source_id > self.target_id:
            source = self.target_id
            target = self.source_id
        else:
            source = self.source_id
            target = self.target_id
            
        key_data = {
            "source_id": source,
            "target_id": target,
            "relationship_type": self.relationship_type
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def add_position(self, start: int, end: int, section: Optional[str] = None) -> None:
        """
        Add a position where this relationship is mentioned in a document.
        
        Args:
            start: Start character position
            end: End character position
            section: Optional section identifier (e.g., "abstract", "introduction")
        """
        position: Dict[str, Any] = {"start": start, "end": end}
        if section:
            position["section"] = section
        self.positions.append(position)
    
    def merge(self, other: 'Relationship') -> 'Relationship':
        """
        Merge another relationship into this one, combining attributes.
        
        Args:
            other: Another relationship to merge with this one
            
        Returns:
            The merged relationship (self)
        """
        # Combine positions
        self.positions.extend(other.positions)
        
        # Update confidence and weight using a weighted average
        total_weight = self.confidence + other.confidence
        self.weight = ((self.weight * self.confidence) + 
                       (other.weight * other.confidence)) / total_weight
        self.confidence = max(self.confidence, other.confidence)
        
        # Update bidirectionality
        self.bidirectional = self.bidirectional or other.bidirectional
        
        # Merge metadata
        for key, value in other.metadata.items():
            if key not in self.metadata:
                self.metadata[key] = value
                
        # If the other relationship has context and this one doesn't, use the other's context
        if not self.context and other.context:
            self.context = other.context
                
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the relationship to a dictionary representation.
        
        Returns:
            Dictionary representation of the relationship
        """
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "weight": self.weight,
            "confidence": self.confidence,
            "bidirectional": self.bidirectional,
            "context": self.context,
            "positions": self.positions,
            "source_document_id": self.source_document_id,
            "metadata": self.metadata,
            "extraction_method": self.extraction_method,
            "fingerprint": self.fingerprint
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """
        Create a Relationship instance from a dictionary.
        
        Args:
            data: Dictionary representation of a relationship
            
        Returns:
            Relationship instance
        """
        # Create a copy to avoid modifying the input
        rel_data = data.copy()
        
        # Handle special fields
        rel_data.pop("fingerprint", None)  # Fingerprint is computed, not stored
        
        # Create the relationship
        return cls(**rel_data)
