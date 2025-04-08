"""
Entity Model for GraphRAG

This module defines the Entity class, which represents named entities and concepts
extracted from documents in the GraphRAG implementation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from uuid import uuid4
import hashlib
import json


@dataclass
class Entity:
    """
    Represents a named entity or concept extracted from a document.
    
    Entities are the nodes in the GraphRAG knowledge graph. They represent
    people, organizations, locations, concepts, or other named elements
    extracted from documents.
    """
    
    # Core attributes
    text: str
    entity_type: str
    
    # Optional attributes with defaults
    id: str = field(default_factory=lambda: str(uuid4()))
    canonical_name: Optional[str] = None
    confidence: float = 1.0
    source_document_id: Optional[str] = None
    
    # Positional information
    positions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Related entities
    aliases: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize derived fields after initialization."""
        if not self.canonical_name:
            self.canonical_name = self.text
            
        # Add the main text as an alias
        self.aliases.add(self.text.lower())
    
    @property
    def fingerprint(self) -> str:
        """
        Generate a unique fingerprint for this entity based on its core attributes.
        
        This is useful for deduplication and identification across documents.
        """
        # Create a canonical name key with null check
        canonical_name_key = ""
        if self.canonical_name is not None:
            canonical_name_key = self.canonical_name.lower()
        
        key_data = {
            "canonical_name": canonical_name_key,
            "entity_type": self.entity_type if self.entity_type is not None else ""
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def add_position(self, start: int, end: int, section: Optional[str] = None) -> None:
        """
        Add a position where this entity appears in a document.
        
        Args:
            start: Start character position
            end: End character position
            section: Optional section identifier (e.g., "abstract", "introduction")
        """
        position: Dict[str, Any] = {"start": start, "end": end}
        if section:
            position["section"] = section
        self.positions.append(position)
    
    def add_alias(self, alias: str) -> None:
        """
        Add an alternative name for this entity.
        
        Args:
            alias: Alternative text representation of this entity
        """
        self.aliases.add(alias.lower())
    
    def merge(self, other: 'Entity') -> 'Entity':
        """
        Merge another entity into this one, combining attributes.
        
        Args:
            other: Another entity to merge with this one
            
        Returns:
            The merged entity (self)
        """
        # Combine aliases
        self.aliases.update(other.aliases)
        
        # Combine positions
        self.positions.extend(other.positions)
        
        # Update confidence if the other entity has higher confidence
        if other.confidence > self.confidence:
            self.confidence = other.confidence
            
        # Merge metadata
        for key, value in other.metadata.items():
            if key not in self.metadata:
                self.metadata[key] = value
                
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entity to a dictionary representation.
        
        Returns:
            Dictionary representation of the entity
        """
        return {
            "id": self.id,
            "text": self.text,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "positions": self.positions,
            "source_document_id": self.source_document_id,
            "metadata": self.metadata,
            "aliases": list(self.aliases),
            "fingerprint": self.fingerprint
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """
        Create an Entity instance from a dictionary.
        
        Args:
            data: Dictionary representation of an entity
            
        Returns:
            Entity instance
        """
        # Create a copy to avoid modifying the input
        entity_data = data.copy()
        
        # Handle special fields
        aliases = set(entity_data.pop("aliases", []))
        entity_data.pop("fingerprint", None)  # Fingerprint is computed, not stored
        
        # Create the entity
        entity = cls(**entity_data)
        
        # Add aliases
        entity.aliases = aliases
        
        return entity
