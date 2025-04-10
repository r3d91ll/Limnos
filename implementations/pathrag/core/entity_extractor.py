"""
Entity Extractor for PathRAG

This module provides entity extraction capabilities for the PathRAG implementation.
It identifies entities from document content that will serve as nodes in the path graph.
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple, TYPE_CHECKING, cast, Union
import logging

# Conditional import for type checking
if TYPE_CHECKING:
    # Define minimal stubs for type checking
    class SpacyDoc:
        """Stub for spaCy Doc"""
        ents: List[Any]
        noun_chunks: List[Any]
        
    class SpacyLanguage:
        """Stub for spaCy Language"""
        pipe_names: List[str]
        
        def add_pipe(self, name: str, *, before: Optional[str] = None) -> Any: ...
        def __call__(self, text: str) -> SpacyDoc: ...
    
    # Mock the spacy module for type checking
    class SpacyModule:
        """Stub for spaCy module"""
        @staticmethod
        def load(model_name: str) -> SpacyLanguage: ...
    
    spacy = SpacyModule()
else:
    import spacy

# Configure logging
logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Extracts entities from document content to serve as nodes in the PathRAG graph.
    
    This extractor identifies named entities, key concepts, and domain-specific terms
    that form the foundation of path-based retrieval.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the entity extractor with configuration.
        
        Args:
            config: Configuration dictionary with options for entity extraction
        """
        self.config = config or {}
        # Using conditional type for self.nlp 
        if TYPE_CHECKING:
            self.nlp: Optional['SpacyLanguage'] = None
        else:
            self.nlp = None
        self.initialized = False
        
        # Entity type mapping for consistent representation
        self.entity_type_map = {
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "PRODUCT": "PRODUCT",
            "DATE": "TIME",
            "TIME": "TIME",
            "MONEY": "QUANTITY",
            "PERCENT": "QUANTITY",
            "QUANTITY": "QUANTITY",
            "NORP": "GROUP",  # Nationalities, religious or political groups
            "FAC": "LOCATION",  # Facilities, buildings, airports
            "WORK_OF_ART": "CREATIVE_WORK",
            "LAW": "CONCEPT",
            "LANGUAGE": "CONCEPT",
            "EVENT": "EVENT"
        }
        
        # Custom entity patterns (can be extended through config)
        self.custom_patterns = self.config.get("custom_patterns", [])
        
        # Stopwords to exclude from entity extraction (common words unlikely to be entities)
        self.stopwords = set(self.config.get("stopwords", []))
        
        # Minimum entity occurrences for inclusion
        self.min_occurrences = self.config.get("min_occurrences", 1)
        
        # Whether to merge adjacent entities if they appear together frequently
        self.merge_adjacent = self.config.get("merge_adjacent", True)
        
        # Maximum allowed tokens for an entity
        self.max_entity_tokens = self.config.get("max_entity_tokens", 6)
    
    def initialize(self) -> None:
        """Initialize the NLP pipeline for entity extraction."""
        # Load appropriate spaCy model based on config
        model_name = self.config.get("spacy_model", "en_core_web_sm")
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"Could not load spaCy model: {model_name}. Downloading...")
            import sys
            import subprocess
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)
        
        # Add custom entity patterns if provided
        # Add null checking to prevent attribute errors
        if self.nlp is not None and self.custom_patterns and "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(self.custom_patterns)
            logger.info(f"Added {len(self.custom_patterns)} custom entity patterns")
        
        self.initialized = True
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from the given text.
        
        Args:
            text: The text content to extract entities from
            
        Returns:
            List of extracted entity dictionaries with metadata
        """
        if not self.initialized:
            self.initialize()
        
        # Process the text with spaCy if nlp is initialized
        if self.nlp is None:
            logger.error("NLP pipeline not initialized")
            return []
            
        doc = self.nlp(text)
        
        # Extract named entities
        entities = []
        entity_counts: Dict[Tuple[str, str], int] = {}
        
        # Process named entities from spaCy's NER
        for ent in doc.ents:
            # Skip entities that are too long
            if len(ent) > self.max_entity_tokens:
                continue
                
            # Skip stopwords
            if ent.text.lower() in self.stopwords:
                continue
            
            # Map entity type to our standardized types
            entity_type = self.entity_type_map.get(ent.label_, "MISC")
            
            entity = {
                "text": ent.text,
                "type": entity_type,
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "ner"
            }
            
            # Count occurrences for filtering later
            entity_key = (ent.text, entity_type)
            entity_counts[entity_key] = entity_counts.get(entity_key, 0) + 1
            
            entities.append(entity)
        
        # Extract noun phrases as potential entities not caught by NER
        if self.config.get("include_noun_phrases", True):
            for chunk in doc.noun_chunks:
                # Skip if too long or too short
                if len(chunk) > self.max_entity_tokens or len(chunk) < 2:
                    continue
                
                # Skip if it's already extracted as a named entity
                already_extracted = any(
                    e["start"] <= chunk.start_char and e["end"] >= chunk.end_char 
                    for e in entities
                )
                
                if not already_extracted and chunk.text.lower() not in self.stopwords:
                    entity = {
                        "text": chunk.text,
                        "type": "CONCEPT",
                        "start": chunk.start_char,
                        "end": chunk.end_char,
                        "source": "noun_chunk"
                    }
                    
                    entity_key = (chunk.text, "CONCEPT")
                    entity_counts[entity_key] = entity_counts.get(entity_key, 0) + 1
                    
                    entities.append(entity)
        
        # Filter entities by minimum occurrences
        if self.min_occurrences > 1:
            entities = [
                e for e in entities 
                if entity_counts.get((e["text"], e["type"]), 0) >= self.min_occurrences
            ]
        
        # Merge adjacent entities if they appear together frequently
        if self.merge_adjacent:
            entities = self._merge_adjacent_entities(entities, text)
        
        # Add unique IDs to entities
        for i, entity in enumerate(entities):
            entity["id"] = f"ent_{i}_{hash(entity['text']) % 10000}"
        
        return entities
    
    def _merge_adjacent_entities(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """
        Merge adjacent entities that frequently appear together.
        
        Args:
            entities: List of extracted entities
            text: Original text for context
            
        Returns:
            List of entities with adjacent ones potentially merged
        """
        # Sort entities by their position in text
        sorted_entities = sorted(entities, key=lambda e: e["start"])
        
        # Look for adjacent entities
        merged_entities = []
        skip_indices = set()
        
        for i in range(len(sorted_entities) - 1):
            if i in skip_indices:
                continue
                
            curr_entity = sorted_entities[i]
            next_entity = sorted_entities[i + 1]
            
            # Check if entities are adjacent (with potential minor gap)
            max_gap = self.config.get("max_merge_gap", 3)  # Maximum character gap to consider entities adjacent
            if 0 <= next_entity["start"] - curr_entity["end"] <= max_gap:
                # Get text between entities
                gap_text = text[curr_entity["end"]:next_entity["start"]]
                
                # If gap is just whitespace or common joining words, merge them
                if gap_text.strip() in ["", "of", "the", "and", "or", "'s"]:
                    merged_text = text[curr_entity["start"]:next_entity["end"]]
                    
                    # Create merged entity
                    merged_entity = {
                        "text": merged_text,
                        "type": curr_entity["type"],  # Use type of first entity
                        "start": curr_entity["start"],
                        "end": next_entity["end"],
                        "source": "merged",
                        "components": [curr_entity, next_entity]
                    }
                    
                    merged_entities.append(merged_entity)
                    skip_indices.add(i)
                    skip_indices.add(i + 1)
                    continue
            
            # If not merged and not already processed
            if i not in skip_indices:
                merged_entities.append(curr_entity)
        
        # Add the last entity if it wasn't merged
        if len(sorted_entities) > 0 and len(sorted_entities) - 1 not in skip_indices:
            merged_entities.append(sorted_entities[-1])
        
        return merged_entities
    
    def get_entity_types(self) -> List[str]:
        """
        Get all entity types supported by this extractor.
        
        Returns:
            List of entity type strings
        """
        return list(set(self.entity_type_map.values()))

    def extract_document_entities(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract entities from a document dictionary.
        
        Args:
            document: Document dictionary with 'content' field containing text
            
        Returns:
            List of extracted entity dictionaries with metadata
        """
        if not document or "content" not in document:
            logger.warning("Document missing 'content' field")
            return []
            
        # Extract entities from document content
        entities = self.extract_entities(document["content"])
        
        # Add document reference to each entity
        for entity in entities:
            entity["document_id"] = document.get("id", "unknown")
            entity["document_title"] = document.get("title", "")
            
        return entities
