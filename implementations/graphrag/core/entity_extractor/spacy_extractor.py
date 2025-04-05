"""
SpaCy-based Entity Extractor for GraphRAG

This module implements entity extraction using the spaCy NLP library.
It provides comprehensive named entity recognition capabilities for GraphRAG.
"""

import spacy
from typing import Dict, List, Optional, Any, Set, Tuple
import logging
from collections import defaultdict

from ..models.entity import Entity
from .entity_extractor import EntityExtractor


class SpacyEntityExtractor(EntityExtractor):
    """
    Entity extractor implementation using spaCy NLP.
    
    This extractor uses spaCy's named entity recognition capabilities to
    identify entities in text. It supports customization of entity types,
    confidence thresholds, and model selection.
    """
    
    # Default entity type mapping from spaCy to GraphRAG types
    DEFAULT_ENTITY_TYPE_MAPPING = {
        "PERSON": "person",
        "ORG": "organization",
        "GPE": "location",
        "LOC": "location",
        "PRODUCT": "product",
        "EVENT": "event",
        "WORK_OF_ART": "work",
        "LAW": "law",
        "DATE": "date",
        "TIME": "time",
        "MONEY": "money",
        "QUANTITY": "quantity",
        "PERCENT": "percent",
        "CARDINAL": "number",
        "ORDINAL": "number",
        "NORP": "group",  # Nationalities, religious or political groups
        "FAC": "facility",  # Buildings, airports, highways, etc.
        "LANGUAGE": "language"
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SpaCy entity extractor.
        
        Args:
            config: Configuration dictionary with the following options:
                - model_name: spaCy model to use (default: "en_core_web_sm")
                - entity_types: List of entity types to extract (default: all)
                - min_confidence: Minimum confidence threshold (default: 0.0)
                - type_mapping: Custom entity type mapping (default: DEFAULT_ENTITY_TYPE_MAPPING)
                - batch_size: Batch size for processing (default: 1000)
                - use_gpu: Whether to use GPU acceleration (default: False)
        """
        super().__init__(config)
        
        # Get configuration options
        self.model_name = self.config.get("model_name", "en_core_web_sm")
        self.entity_types = set(self.config.get("entity_types", []))
        self.min_confidence = self.config.get("min_confidence", 0.0)
        self.batch_size = self.config.get("batch_size", 1000)
        self.use_gpu = self.config.get("use_gpu", False)
        
        # Set up entity type mapping
        self.type_mapping = self.DEFAULT_ENTITY_TYPE_MAPPING.copy()
        custom_mapping = self.config.get("type_mapping", {})
        self.type_mapping.update(custom_mapping)
        
        # Initialize spaCy model
        self.logger.info(f"Loading spaCy model: {self.model_name}")
        try:
            self.nlp = spacy.load(self.model_name)
            
            # Configure GPU if requested
            if self.use_gpu:
                spacy.prefer_gpu()
                self.logger.info("Using GPU for spaCy processing")
                
            self.logger.info(f"Successfully loaded spaCy model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load spaCy model: {str(e)}")
            raise
    
    def extract_entities(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """
        Extract entities from the given text using spaCy.
        
        Args:
            text: The text to extract entities from
            metadata: Optional metadata about the text
            
        Returns:
            List of extracted entities
        """
        if not text or not text.strip():
            return []
            
        # Process the text with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            # Map spaCy entity type to GraphRAG type
            entity_type = self.type_mapping.get(ent.label_, "unknown")
            
            # Skip if we're filtering by entity type and this type is not included
            if self.entity_types and entity_type not in self.entity_types:
                continue
                
            # Create the entity
            entity = Entity(
                text=ent.text,
                entity_type=entity_type,
                confidence=1.0,  # spaCy doesn't provide confidence scores by default
                metadata=metadata.copy() if metadata else {}
            )
            
            # Add position information
            entity.add_position(ent.start_char, ent.end_char)
            
            entities.append(entity)
            
        return entities
    
    def extract_entities_with_context(self, text: str, 
                                    context_window: int = 50,
                                    metadata: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """
        Extract entities with surrounding context.
        
        Args:
            text: The text to extract entities from
            context_window: Number of characters to include as context around each entity
            metadata: Optional metadata about the text
            
        Returns:
            List of extracted entities with context in metadata
        """
        entities = self.extract_entities(text, metadata)
        
        # Add context to each entity
        for entity in entities:
            for pos in entity.positions:
                start = pos["start"]
                end = pos["end"]
                
                # Calculate context boundaries
                context_start = max(0, start - context_window)
                context_end = min(len(text), end + context_window)
                
                # Extract context
                context = text[context_start:context_end]
                
                # Store context in metadata
                if "contexts" not in entity.metadata:
                    entity.metadata["contexts"] = []
                    
                entity.metadata["contexts"].append(context)
                
        return entities
    
    def extract_noun_phrases(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """
        Extract noun phrases as concept entities.
        
        Args:
            text: The text to extract noun phrases from
            metadata: Optional metadata about the text
            
        Returns:
            List of extracted concept entities
        """
        if not text or not text.strip():
            return []
            
        # Process the text with spaCy
        doc = self.nlp(text)
        
        # Extract noun phrases
        entities = []
        for chunk in doc.noun_chunks:
            # Create the entity
            entity = Entity(
                text=chunk.text,
                entity_type="concept",
                confidence=0.8,  # Lower confidence for noun phrases
                metadata=metadata.copy() if metadata else {}
            )
            
            # Add position information
            entity.add_position(chunk.start_char, chunk.end_char)
            
            entities.append(entity)
            
        return entities
    
    def extract_entities_batch(self, texts: List[str], 
                             metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[List[Entity]]:
        """
        Extract entities from multiple texts in batch mode.
        
        Args:
            texts: List of texts to process
            metadata_list: Optional list of metadata dictionaries (one per text)
            
        Returns:
            List of lists of extracted entities (one list per input text)
        """
        if not texts:
            return []
            
        # Ensure metadata list matches texts length
        if metadata_list is None:
            metadata_list = [None] * len(texts)
        elif len(metadata_list) != len(texts):
            raise ValueError("metadata_list length must match texts length")
            
        # Process texts in batches
        all_entities = []
        for doc, metadata in zip(self.nlp.pipe(texts, batch_size=self.batch_size), metadata_list):
            # Extract entities from this document
            entities = []
            for ent in doc.ents:
                # Map spaCy entity type to GraphRAG type
                entity_type = self.type_mapping.get(ent.label_, "unknown")
                
                # Skip if we're filtering by entity type and this type is not included
                if self.entity_types and entity_type not in self.entity_types:
                    continue
                    
                # Create the entity
                entity = Entity(
                    text=ent.text,
                    entity_type=entity_type,
                    confidence=1.0,
                    metadata=metadata.copy() if metadata else {}
                )
                
                # Add position information
                entity.add_position(ent.start_char, ent.end_char)
                
                entities.append(entity)
                
            all_entities.append(entities)
            
        return all_entities
    
    def enrich_entities(self, entities: List[Entity], text: str) -> List[Entity]:
        """
        Enrich entities with additional information from spaCy analysis.
        
        Args:
            entities: List of entities to enrich
            text: The original text
            
        Returns:
            Enriched entities
        """
        if not entities or not text:
            return entities
            
        # Process the text with spaCy
        doc = self.nlp(text)
        
        # Create a mapping of spans to entities
        span_to_entity = {}
        for entity in entities:
            for pos in entity.positions:
                span = (pos["start"], pos["end"])
                span_to_entity[span] = entity
                
        # Enrich entities with linguistic information
        for token in doc:
            # Find entities that contain this token
            for span, entity in span_to_entity.items():
                start, end = span
                if start <= token.idx < end:
                    # Add part of speech information
                    if "pos_tags" not in entity.metadata:
                        entity.metadata["pos_tags"] = []
                    entity.metadata["pos_tags"].append(token.pos_)
                    
                    # Add lemma information
                    if "lemmas" not in entity.metadata:
                        entity.metadata["lemmas"] = []
                    entity.metadata["lemmas"].append(token.lemma_)
                    
        return entities
