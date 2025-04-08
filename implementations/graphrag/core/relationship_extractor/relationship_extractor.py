"""
Base Relationship Extractor for GraphRAG

This module defines the base RelationshipExtractor class that provides the interface
for all relationship extraction implementations in GraphRAG.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set, Tuple
import logging
from dataclasses import dataclass

from ..models.entity import Entity
from ..models.relationship import Relationship
from ..models.document import DocumentReference


class RelationshipExtractor(ABC):
    """
    Abstract base class for relationship extractors in GraphRAG.
    
    Relationship extractors are responsible for identifying connections between
    entities in documents. They form the edges of the knowledge graph, connecting
    entity nodes to represent semantic relationships.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the relationship extractor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def extract_relationships(self, text: str, entities: List[Entity], 
                             metadata: Optional[Dict[str, Any]] = None) -> List[Relationship]:
        """
        Extract relationships between entities from the given text.
        
        Args:
            text: The text to extract relationships from
            entities: List of entities to find relationships between
            metadata: Optional metadata about the text (e.g., source, section)
            
        Returns:
            List of extracted relationships
        """
        pass
    
    def process_document(self, document: DocumentReference, 
                        content: str, 
                        entities: List[Entity],
                        metadata: Optional[Dict[str, Any]] = None) -> List[Relationship]:
        """
        Process a document to extract relationships between entities.
        
        Args:
            document: Reference to the document being processed
            content: The document content
            entities: List of entities to find relationships between
            metadata: Optional additional metadata
            
        Returns:
            List of extracted relationships
        """
        self.logger.info(f"Processing document: {document.document_id}")
        
        # Combine document metadata with provided metadata
        combined_metadata = {
            "document_id": document.document_id,
            "document_type": document.document_type,
            "title": document.title
        }
        
        if metadata:
            combined_metadata.update(metadata)
            
        # Extract relationships
        relationships = self.extract_relationships(content, entities, combined_metadata)
        
        # Set source document ID for all relationships
        for relationship in relationships:
            relationship.source_document_id = document.document_id
            
        self.logger.info(f"Extracted {len(relationships)} relationships from document {document.document_id}")
        return relationships
    
    def batch_process(self, documents: List[Tuple[DocumentReference, str, List[Entity], Optional[Dict[str, Any]]]]) -> Dict[str, List[Relationship]]:
        """
        Process multiple documents to extract relationships.
        
        Args:
            documents: List of (document_reference, content, entities, metadata) tuples
            
        Returns:
            Dictionary mapping document IDs to lists of extracted relationships
        """
        self.logger.info(f"Batch processing {len(documents)} documents")
        results = {}
        
        for doc_ref, content, entities, metadata in documents:
            relationships = self.process_document(doc_ref, content, entities, metadata)
            results[doc_ref.document_id] = relationships
            
        return results
    
    def filter_relationships(self, relationships: List[Relationship], 
                           min_confidence: float = 0.0,
                           relationship_types: Optional[Set[str]] = None) -> List[Relationship]:
        """
        Filter relationships based on confidence and type.
        
        Args:
            relationships: List of relationships to filter
            min_confidence: Minimum confidence threshold
            relationship_types: Set of relationship types to include (None for all)
            
        Returns:
            Filtered list of relationships
        """
        filtered = []
        
        for relationship in relationships:
            # Check confidence threshold
            if relationship.confidence < min_confidence:
                continue
                
            # Check relationship type if specified
            if relationship_types and relationship.relationship_type not in relationship_types:
                continue
                
            filtered.append(relationship)
            
        return filtered
    
    def deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """
        Deduplicate relationships based on their fingerprints.
        
        Args:
            relationships: List of relationships to deduplicate
            
        Returns:
            Deduplicated list of relationships
        """
        unique_relationships: Dict[str, Relationship] = {}
        
        for relationship in relationships:
            fingerprint = relationship.fingerprint
            
            if fingerprint in unique_relationships:
                # Merge with existing relationship
                unique_relationships[fingerprint].merge(relationship)
            else:
                # Add as new relationship
                unique_relationships[fingerprint] = relationship
                
        return list(unique_relationships.values())
