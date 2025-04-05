"""
Entity Extractor Factory for GraphRAG

This module provides a factory class for creating and configuring entity extractors
in the GraphRAG implementation.
"""

from typing import Dict, List, Optional, Any, Type
import logging

from .entity_extractor import EntityExtractor
from .spacy_extractor import SpacyEntityExtractor
from .academic_extractor import AcademicEntityExtractor


class EntityExtractorFactory:
    """
    Factory class for creating entity extractors.
    
    This factory provides a unified interface for creating and configuring
    different types of entity extractors based on document type, configuration,
    and other parameters.
    """
    
    # Registry of available extractors
    EXTRACTORS = {
        "spacy": SpacyEntityExtractor,
        "academic": AcademicEntityExtractor,
        # Add more extractors as they are implemented
    }
    
    @classmethod
    def create(cls, extractor_type: str, config: Optional[Dict[str, Any]] = None) -> EntityExtractor:
        """
        Create an entity extractor of the specified type.
        
        Args:
            extractor_type: Type of extractor to create
            config: Optional configuration for the extractor
            
        Returns:
            Configured entity extractor instance
            
        Raises:
            ValueError: If the specified extractor type is not registered
        """
        logger = logging.getLogger(f"{__name__}.EntityExtractorFactory")
        
        # Check if the extractor type is registered
        if extractor_type not in cls.EXTRACTORS:
            registered = ", ".join(cls.EXTRACTORS.keys())
            logger.error(f"Unknown extractor type: {extractor_type}. Registered types: {registered}")
            raise ValueError(f"Unknown extractor type: {extractor_type}")
            
        # Get the extractor class
        extractor_class = cls.EXTRACTORS[extractor_type]
        logger.info(f"Creating {extractor_type} entity extractor")
        
        # Create and return the extractor
        return extractor_class(config)
    
    @classmethod
    def create_for_document_type(cls, document_type: str, 
                                config: Optional[Dict[str, Any]] = None) -> EntityExtractor:
        """
        Create an appropriate entity extractor for the specified document type.
        
        Args:
            document_type: Type of document (e.g., "academic", "web", "code")
            config: Optional configuration for the extractor
            
        Returns:
            Configured entity extractor instance suitable for the document type
        """
        logger = logging.getLogger(f"{__name__}.EntityExtractorFactory")
        logger.info(f"Creating entity extractor for document type: {document_type}")
        
        # Map document types to appropriate extractors
        document_type_mapping = {
            "academic": "academic",
            "paper": "academic",
            "research": "academic",
            "article": "academic",
            "web": "spacy",
            "text": "spacy",
            "news": "spacy",
            "code": "spacy",  # For now, use spaCy for code documents
            "default": "spacy"
        }
        
        # Get the appropriate extractor type
        extractor_type = document_type_mapping.get(document_type.lower(), "default")
        if extractor_type == "default":
            logger.info(f"No specific extractor for document type '{document_type}', using default")
            
        # Create and return the extractor
        return cls.create(extractor_type, config)
    
    @classmethod
    def register_extractor(cls, name: str, extractor_class: Type[EntityExtractor]) -> None:
        """
        Register a new entity extractor type.
        
        Args:
            name: Name to register the extractor under
            extractor_class: The extractor class to register
            
        Raises:
            TypeError: If the extractor class is not a subclass of EntityExtractor
        """
        logger = logging.getLogger(f"{__name__}.EntityExtractorFactory")
        
        # Check that the class is a subclass of EntityExtractor
        if not issubclass(extractor_class, EntityExtractor):
            logger.error(f"Cannot register {extractor_class.__name__}: not a subclass of EntityExtractor")
            raise TypeError(f"Extractor class must be a subclass of EntityExtractor")
            
        # Register the extractor
        cls.EXTRACTORS[name] = extractor_class
        logger.info(f"Registered entity extractor: {name}")
