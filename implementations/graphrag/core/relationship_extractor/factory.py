"""
Relationship Extractor Factory for GraphRAG

This module provides a factory class for creating and configuring relationship extractors
in the GraphRAG implementation.
"""

from typing import Dict, List, Optional, Any, Type
import logging

from .relationship_extractor import RelationshipExtractor
from .cooccurrence_extractor import CooccurrenceRelationshipExtractor
from .dependency_extractor import DependencyRelationshipExtractor
from .citation_extractor import CitationRelationshipExtractor


class RelationshipExtractorFactory:
    """
    Factory class for creating relationship extractors.
    
    This factory provides a unified interface for creating and configuring
    different types of relationship extractors based on document type, configuration,
    and other parameters.
    """
    
    # Registry of available extractors
    EXTRACTORS = {
        "cooccurrence": CooccurrenceRelationshipExtractor,
        "dependency": DependencyRelationshipExtractor,
        "citation": CitationRelationshipExtractor,
        # Add more extractors as they are implemented
    }
    
    @classmethod
    def create(cls, extractor_type: str, config: Optional[Dict[str, Any]] = None) -> RelationshipExtractor:
        """
        Create a relationship extractor of the specified type.
        
        Args:
            extractor_type: Type of extractor to create
            config: Optional configuration for the extractor
            
        Returns:
            Configured relationship extractor instance
            
        Raises:
            ValueError: If the specified extractor type is not registered
        """
        logger = logging.getLogger(f"{__name__}.RelationshipExtractorFactory")
        
        # Check if the extractor type is registered
        if extractor_type not in cls.EXTRACTORS:
            registered = ", ".join(cls.EXTRACTORS.keys())
            logger.error(f"Unknown extractor type: {extractor_type}. Registered types: {registered}")
            raise ValueError(f"Unknown extractor type: {extractor_type}")
            
        # Get the extractor class
        extractor_class = cls.EXTRACTORS[extractor_type]
        logger.info(f"Creating {extractor_type} relationship extractor")
        
        # Create and return the extractor
        return extractor_class(config)
    
    @classmethod
    def create_for_document_type(cls, document_type: str, 
                                config: Optional[Dict[str, Any]] = None) -> List[RelationshipExtractor]:
        """
        Create appropriate relationship extractors for the specified document type.
        
        Args:
            document_type: Type of document (e.g., "academic", "web", "code")
            config: Optional configuration for the extractors
            
        Returns:
            List of configured relationship extractor instances suitable for the document type
        """
        logger = logging.getLogger(f"{__name__}.RelationshipExtractorFactory")
        logger.info(f"Creating relationship extractors for document type: {document_type}")
        
        # Map document types to appropriate extractors
        document_type_mapping = {
            "academic": ["cooccurrence", "dependency", "citation"],
            "paper": ["cooccurrence", "dependency", "citation"],
            "research": ["cooccurrence", "dependency", "citation"],
            "article": ["cooccurrence", "dependency"],
            "web": ["cooccurrence", "dependency"],
            "text": ["cooccurrence", "dependency"],
            "news": ["cooccurrence", "dependency"],
            "code": ["cooccurrence"],  # For now, just use co-occurrence for code
            "default": ["cooccurrence"]
        }
        
        # Get the appropriate extractor types
        extractor_types = document_type_mapping.get(document_type.lower(), document_type_mapping["default"])
        
        # Create and return the extractors
        extractors = []
        for extractor_type in extractor_types:
            # Create a copy of the config for each extractor
            extractor_config = config.copy() if config else {}
            extractors.append(cls.create(extractor_type, extractor_config))
            
        return extractors
    
    @classmethod
    def create_composite_extractor(cls, document_type: str, 
                                 config: Optional[Dict[str, Any]] = None) -> 'CompositeRelationshipExtractor':
        """
        Create a composite relationship extractor for the specified document type.
        
        Args:
            document_type: Type of document (e.g., "academic", "web", "code")
            config: Optional configuration for the extractors
            
        Returns:
            CompositeRelationshipExtractor containing appropriate extractors for the document type
        """
        # Create individual extractors
        extractors = cls.create_for_document_type(document_type, config)
        
        # Create and return a composite extractor
        return CompositeRelationshipExtractor(extractors, config)
    
    @classmethod
    def register_extractor(cls, name: str, extractor_class: Type[RelationshipExtractor]) -> None:
        """
        Register a new relationship extractor type.
        
        Args:
            name: Name to register the extractor under
            extractor_class: The extractor class to register
            
        Raises:
            TypeError: If the extractor class is not a subclass of RelationshipExtractor
        """
        logger = logging.getLogger(f"{__name__}.RelationshipExtractorFactory")
        
        # Check that the class is a subclass of RelationshipExtractor
        if not issubclass(extractor_class, RelationshipExtractor):
            logger.error(f"Cannot register {extractor_class.__name__}: not a subclass of RelationshipExtractor")
            raise TypeError(f"Extractor class must be a subclass of RelationshipExtractor")
            
        # Register the extractor
        cls.EXTRACTORS[name] = extractor_class
        logger.info(f"Registered relationship extractor: {name}")


class CompositeRelationshipExtractor(RelationshipExtractor):
    """
    Composite relationship extractor that combines multiple extractors.
    
    This extractor delegates relationship extraction to multiple underlying
    extractors and combines their results. It provides a unified interface
    for using multiple extraction strategies simultaneously.
    """
    
    def __init__(self, extractors: List[RelationshipExtractor], 
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the composite relationship extractor.
        
        Args:
            extractors: List of relationship extractors to use
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.extractors = extractors
        self.logger.info(f"Created composite extractor with {len(extractors)} extractors")
    
    def extract_relationships(self, text: str, entities: List[Entity], 
                             metadata: Optional[Dict[str, Any]] = None) -> List[Relationship]:
        """
        Extract relationships using all underlying extractors.
        
        Args:
            text: The text to extract relationships from
            entities: List of entities to find relationships between
            metadata: Optional metadata about the text
            
        Returns:
            Combined list of extracted relationships
        """
        all_relationships = []
        
        # Use each extractor
        for extractor in self.extractors:
            extractor_name = extractor.__class__.__name__
            self.logger.debug(f"Using {extractor_name} to extract relationships")
            
            # Extract relationships
            relationships = extractor.extract_relationships(text, entities, metadata)
            self.logger.debug(f"{extractor_name} found {len(relationships)} relationships")
            
            # Add to combined results
            all_relationships.extend(relationships)
            
        # Deduplicate relationships
        deduplicated = self.deduplicate_relationships(all_relationships)
        self.logger.info(f"Extracted {len(deduplicated)} unique relationships from all extractors")
        
        return deduplicated
