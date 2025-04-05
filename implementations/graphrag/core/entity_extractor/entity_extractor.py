"""
Base Entity Extractor for GraphRAG

This module defines the base EntityExtractor class that provides the interface
for all entity extraction implementations in GraphRAG.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set, Tuple
import logging
from dataclasses import dataclass

from ..models.entity import Entity
from ..models.document import DocumentReference


class EntityExtractor(ABC):
    """
    Abstract base class for entity extractors in GraphRAG.
    
    Entity extractors are responsible for identifying named entities and concepts
    in documents. They form the foundation of the graph construction process by
    providing the nodes that will be connected by relationships.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the entity extractor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def extract_entities(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """
        Extract entities from the given text.
        
        Args:
            text: The text to extract entities from
            metadata: Optional metadata about the text (e.g., source, section)
            
        Returns:
            List of extracted entities
        """
        pass
    
    def process_document(self, document: DocumentReference, 
                        content: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """
        Process a document to extract entities.
        
        Args:
            document: Reference to the document being processed
            content: The document content
            metadata: Optional additional metadata
            
        Returns:
            List of extracted entities
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
            
        # Extract entities
        entities = self.extract_entities(content, combined_metadata)
        
        # Set source document ID for all entities
        for entity in entities:
            entity.source_document_id = document.document_id
            
        self.logger.info(f"Extracted {len(entities)} entities from document {document.document_id}")
        return entities
    
    def batch_process(self, documents: List[Tuple[DocumentReference, str, Optional[Dict[str, Any]]]]) -> Dict[str, List[Entity]]:
        """
        Process multiple documents to extract entities.
        
        Args:
            documents: List of (document_reference, content, metadata) tuples
            
        Returns:
            Dictionary mapping document IDs to lists of extracted entities
        """
        self.logger.info(f"Batch processing {len(documents)} documents")
        results = {}
        
        for doc_ref, content, metadata in documents:
            entities = self.process_document(doc_ref, content, metadata)
            results[doc_ref.document_id] = entities
            
        return results
    
    def filter_entities(self, entities: List[Entity], 
                       min_confidence: float = 0.0,
                       entity_types: Optional[Set[str]] = None) -> List[Entity]:
        """
        Filter entities based on confidence and type.
        
        Args:
            entities: List of entities to filter
            min_confidence: Minimum confidence threshold
            entity_types: Set of entity types to include (None for all)
            
        Returns:
            Filtered list of entities
        """
        filtered = []
        
        for entity in entities:
            # Check confidence threshold
            if entity.confidence < min_confidence:
                continue
                
            # Check entity type if specified
            if entity_types and entity.entity_type not in entity_types:
                continue
                
            filtered.append(entity)
            
        return filtered
    
    def deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Deduplicate entities based on their fingerprints.
        
        Args:
            entities: List of entities to deduplicate
            
        Returns:
            Deduplicated list of entities
        """
        unique_entities = {}
        
        for entity in entities:
            fingerprint = entity.fingerprint
            
            if fingerprint in unique_entities:
                # Merge with existing entity
                unique_entities[fingerprint].merge(entity)
            else:
                # Add as new entity
                unique_entities[fingerprint] = entity
                
        return list(unique_entities.values())
