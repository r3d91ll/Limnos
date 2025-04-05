"""
Dependency-based Relationship Extractor for GraphRAG

This module implements relationship extraction based on syntactic dependencies
between entities in text, using spaCy's dependency parsing capabilities.
"""

import spacy
from typing import Dict, List, Optional, Any, Set, Tuple
import logging
from collections import defaultdict

from ..models.entity import Entity
from ..models.relationship import Relationship
from .relationship_extractor import RelationshipExtractor


class DependencyRelationshipExtractor(RelationshipExtractor):
    """
    Relationship extractor implementation based on syntactic dependencies.
    
    This extractor identifies explicit relationships between entities based on
    the syntactic structure of sentences. It uses spaCy's dependency parsing
    to identify subject-verb-object patterns and other dependency structures
    that indicate semantic relationships.
    """
    
    # Default relationship mapping for dependency patterns
    DEFAULT_DEPENDENCY_PATTERNS = {
        "nsubj-dobj": "acts_on",
        "nsubj-prep-pobj": "relates_to",
        "nsubj-nsubjpass": "associated_with",
        "compound": "part_of",
        "nmod": "modified_by",
        "amod": "has_property",
        "appos": "also_known_as"
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the dependency relationship extractor.
        
        Args:
            config: Configuration dictionary with the following options:
                - model_name: spaCy model to use (default: "en_core_web_sm")
                - dependency_patterns: Custom dependency pattern mapping
                - min_confidence: Minimum confidence threshold (default: 0.6)
                - use_gpu: Whether to use GPU acceleration (default: False)
                - max_sentence_length: Maximum sentence length to process (default: 200)
                - extract_verbs: Whether to extract verb-based relationships (default: True)
        """
        super().__init__(config)
        
        # Get configuration options
        self.model_name = self.config.get("model_name", "en_core_web_sm")
        self.min_confidence = self.config.get("min_confidence", 0.6)
        self.use_gpu = self.config.get("use_gpu", False)
        self.max_sentence_length = self.config.get("max_sentence_length", 200)
        self.extract_verbs = self.config.get("extract_verbs", True)
        
        # Set up dependency pattern mapping
        self.dependency_patterns = self.DEFAULT_DEPENDENCY_PATTERNS.copy()
        custom_patterns = self.config.get("dependency_patterns", {})
        self.dependency_patterns.update(custom_patterns)
        
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
    
    def extract_relationships(self, text: str, entities: List[Entity], 
                             metadata: Optional[Dict[str, Any]] = None) -> List[Relationship]:
        """
        Extract dependency-based relationships between entities from the given text.
        
        Args:
            text: The text to extract relationships from
            entities: List of entities to find relationships between
            metadata: Optional metadata about the text
            
        Returns:
            List of extracted relationships
        """
        if not text or not entities or len(entities) < 2:
            return []
            
        # Create a mapping of text positions to entities
        position_to_entity = {}
        for entity in entities:
            for position in entity.positions:
                start = position["start"]
                end = position["end"]
                position_to_entity[(start, end)] = entity
                
        # Split text into sentences (simple approach for now)
        sentences = text.split('.')
        
        # Process each sentence to extract relationships
        all_relationships = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) > self.max_sentence_length:
                continue
                
            # Process the sentence with spaCy
            doc = self.nlp(sentence)
            
            # Extract relationships from this sentence
            sentence_relationships = self._extract_from_sentence(doc, position_to_entity, metadata)
            all_relationships.extend(sentence_relationships)
            
        return self.deduplicate_relationships(all_relationships)
    
    def _extract_from_sentence(self, doc, position_to_entity: Dict[Tuple[int, int], Entity],
                              metadata: Optional[Dict[str, Any]] = None) -> List[Relationship]:
        """
        Extract relationships from a spaCy-processed sentence.
        
        Args:
            doc: spaCy Doc object
            position_to_entity: Mapping of text positions to entities
            metadata: Optional metadata
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Map spaCy tokens to entities
        token_to_entity = {}
        for token in doc:
            token_start = token.idx
            token_end = token.idx + len(token.text)
            
            # Check if this token is part of an entity
            for (start, end), entity in position_to_entity.items():
                if (token_start >= start and token_end <= end) or \
                   (start >= token_start and start < token_end) or \
                   (token_start >= start and token_start < end):
                    token_to_entity[token] = entity
                    break
        
        # Extract subject-verb-object patterns
        if self.extract_verbs:
            verb_relationships = self._extract_svo_relationships(doc, token_to_entity, metadata)
            relationships.extend(verb_relationships)
            
        # Extract dependency-based patterns
        dependency_relationships = self._extract_dependency_relationships(doc, token_to_entity, metadata)
        relationships.extend(dependency_relationships)
        
        return relationships
    
    def _extract_svo_relationships(self, doc, token_to_entity: Dict, 
                                 metadata: Optional[Dict[str, Any]] = None) -> List[Relationship]:
        """
        Extract subject-verb-object relationships from a sentence.
        
        Args:
            doc: spaCy Doc object
            token_to_entity: Mapping of tokens to entities
            metadata: Optional metadata
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Find verbs
        for token in doc:
            if token.pos_ == "VERB":
                # Find subjects and objects connected to this verb
                subjects = []
                objects = []
                
                # Check for subjects
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass") and child in token_to_entity:
                        subjects.append(child)
                        
                    # Check for objects
                    if child.dep_ in ("dobj", "pobj", "attr") and child in token_to_entity:
                        objects.append(child)
                        
                    # Check for prepositional phrases
                    if child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj" and grandchild in token_to_entity:
                                objects.append(grandchild)
                
                # Create relationships between subjects and objects
                for subject in subjects:
                    for obj in objects:
                        # Skip if subject and object are the same entity
                        if token_to_entity[subject].id == token_to_entity[obj].id:
                            continue
                            
                        # Create the relationship
                        relationship = Relationship(
                            source_id=token_to_entity[subject].id,
                            target_id=token_to_entity[obj].id,
                            relationship_type=token.lemma_,  # Use the verb lemma as relationship type
                            confidence=0.8,
                            weight=1.0,
                            bidirectional=False,
                            extraction_method="syntactic_dependency"
                        )
                        
                        # Add context
                        relationship.context = doc.text
                        
                        # Add position information
                        relationship.add_position(token.idx, token.idx + len(token.text))
                        
                        # Add metadata
                        if metadata:
                            relationship.metadata.update(metadata)
                            
                        # Add dependency information to metadata
                        relationship.metadata["pattern"] = "svo"
                        relationship.metadata["verb"] = token.text
                        relationship.metadata["subject"] = subject.text
                        relationship.metadata["object"] = obj.text
                        
                        relationships.append(relationship)
                        
        return relationships
    
    def _extract_dependency_relationships(self, doc, token_to_entity: Dict,
                                        metadata: Optional[Dict[str, Any]] = None) -> List[Relationship]:
        """
        Extract relationships based on dependency patterns.
        
        Args:
            doc: spaCy Doc object
            token_to_entity: Mapping of tokens to entities
            metadata: Optional metadata
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Check each token for dependency patterns
        for token in doc:
            # Skip tokens that aren't part of an entity
            if token not in token_to_entity:
                continue
                
            # Check each child of this token
            for child in token.children:
                # Skip children that aren't part of an entity
                if child not in token_to_entity:
                    continue
                    
                # Skip if both tokens are part of the same entity
                if token_to_entity[token].id == token_to_entity[child].id:
                    continue
                    
                # Check if this dependency pattern is in our mapping
                pattern_key = token.dep_ + "-" + child.dep_
                relationship_type = self.dependency_patterns.get(pattern_key)
                
                # If pattern not found, check individual dependencies
                if not relationship_type:
                    relationship_type = self.dependency_patterns.get(child.dep_)
                    
                # If still no match, use a generic relationship type
                if not relationship_type:
                    relationship_type = "related_to"
                    
                # Create the relationship
                relationship = Relationship(
                    source_id=token_to_entity[token].id,
                    target_id=token_to_entity[child].id,
                    relationship_type=relationship_type,
                    confidence=0.7,
                    weight=0.8,
                    bidirectional=False,
                    extraction_method="dependency_pattern"
                )
                
                # Add context
                relationship.context = doc.text
                
                # Add position information
                start = min(token.idx, child.idx)
                end = max(token.idx + len(token.text), child.idx + len(child.text))
                relationship.add_position(start, end)
                
                # Add metadata
                if metadata:
                    relationship.metadata.update(metadata)
                    
                # Add dependency information to metadata
                relationship.metadata["pattern"] = pattern_key
                relationship.metadata["head_dep"] = token.dep_
                relationship.metadata["child_dep"] = child.dep_
                
                relationships.append(relationship)
                
        return relationships
    
    def extract_relationships_batch(self, texts: List[str], 
                                  entities_list: List[List[Entity]],
                                  metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[List[Relationship]]:
        """
        Extract relationships from multiple texts in batch mode.
        
        Args:
            texts: List of texts to process
            entities_list: List of entity lists (one per text)
            metadata_list: Optional list of metadata dictionaries (one per text)
            
        Returns:
            List of lists of extracted relationships (one list per input text)
        """
        if not texts:
            return []
            
        # Ensure metadata list matches texts length
        if metadata_list is None:
            metadata_list = [None] * len(texts)
        elif len(metadata_list) != len(texts):
            raise ValueError("metadata_list length must match texts length")
            
        # Ensure entities list matches texts length
        if len(entities_list) != len(texts):
            raise ValueError("entities_list length must match texts length")
            
        # Process each text
        all_relationships = []
        for text, entities, metadata in zip(texts, entities_list, metadata_list):
            relationships = self.extract_relationships(text, entities, metadata)
            all_relationships.append(relationships)
            
        return all_relationships
