"""
Co-occurrence Relationship Extractor for GraphRAG

This module implements relationship extraction based on entity co-occurrence within
text windows, sentences, or sections in documents.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
import logging
import re
from collections import defaultdict

from ..models.entity import Entity
from ..models.relationship import Relationship
from .relationship_extractor import RelationshipExtractor


class CooccurrenceRelationshipExtractor(RelationshipExtractor):
    """
    Relationship extractor implementation based on entity co-occurrence.
    
    This extractor identifies implicit relationships between entities based on
    their co-occurrence within text windows, sentences, or sections. The strength
    of the relationship is determined by factors such as proximity, frequency,
    and context.
    """
    
    # Default co-occurrence window sizes
    DEFAULT_WINDOW_SIZES = [1, 2, 3]  # Number of sentences
    
    # Default relationship types based on proximity
    DEFAULT_RELATIONSHIP_TYPES = {
        1: "strongly_related",
        2: "related",
        3: "weakly_related"
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the co-occurrence relationship extractor.
        
        Args:
            config: Configuration dictionary with the following options:
                - window_sizes: List of window sizes in sentences (default: [1, 2, 3])
                - relationship_types: Dict mapping window sizes to relationship types
                - min_confidence: Minimum confidence threshold (default: 0.5)
                - max_distance: Maximum character distance for relationship (default: 500)
                - use_sections: Whether to use document sections (default: True)
                - bidirectional: Whether relationships are bidirectional (default: True)
        """
        super().__init__(config)
        
        # Get configuration options
        self.window_sizes = self.config.get("window_sizes", self.DEFAULT_WINDOW_SIZES)
        self.min_confidence = self.config.get("min_confidence", 0.5)
        self.max_distance = self.config.get("max_distance", 500)
        self.use_sections = self.config.get("use_sections", True)
        self.bidirectional = self.config.get("bidirectional", True)
        
        # Set up relationship type mapping
        self.relationship_types = self.DEFAULT_RELATIONSHIP_TYPES.copy()
        custom_types = self.config.get("relationship_types", {})
        self.relationship_types.update(custom_types)
        
        # Compile sentence splitting regex
        self.sentence_splitter = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
    
    def extract_relationships(self, text: str, entities: List[Entity], 
                             metadata: Optional[Dict[str, Any]] = None) -> List[Relationship]:
        """
        Extract co-occurrence relationships between entities from the given text.
        
        Args:
            text: The text to extract relationships from
            entities: List of entities to find relationships between
            metadata: Optional metadata about the text
            
        Returns:
            List of extracted relationships
        """
        if not text or not entities or len(entities) < 2:
            return []
            
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        # Get entity positions by sentence
        entity_positions = self._map_entities_to_sentences(entities, sentences, text)
        
        # Extract relationships based on co-occurrence
        relationships = []
        
        # Process each window size
        for window_size in self.window_sizes:
            window_relationships = self._extract_relationships_with_window(
                entities, entity_positions, window_size, sentences, metadata
            )
            relationships.extend(window_relationships)
            
        # Deduplicate relationships
        return self.deduplicate_relationships(relationships)
    
    def _split_into_sentences(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into sentences and track their positions.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentence dictionaries with text and position information
        """
        # Split text into sentences
        sentence_spans = []
        last_end = 0
        
        for match in self.sentence_splitter.finditer(text):
            end = match.end()
            sentence_text = text[last_end:end].strip()
            if sentence_text:
                sentence_spans.append({
                    "text": sentence_text,
                    "start": last_end,
                    "end": end
                })
            last_end = end
            
        # Add the last sentence if there is one
        if last_end < len(text):
            sentence_text = text[last_end:].strip()
            if sentence_text:
                sentence_spans.append({
                    "text": sentence_text,
                    "start": last_end,
                    "end": len(text)
                })
                
        return sentence_spans
    
    def _map_entities_to_sentences(self, entities: List[Entity], 
                                  sentences: List[Dict[str, Any]],
                                  text: str) -> Dict[int, List[Tuple[Entity, int]]]:
        """
        Map entities to the sentences they appear in.
        
        Args:
            entities: List of entities
            sentences: List of sentence dictionaries
            text: Original text
            
        Returns:
            Dictionary mapping sentence indices to lists of (entity, position_index) tuples
        """
        entity_positions = defaultdict(list)
        
        # For each entity, find which sentences it appears in
        for entity in entities:
            for pos_idx, position in enumerate(entity.positions):
                entity_start = position["start"]
                entity_end = position["end"]
                
                # Find which sentence(s) this position falls into
                for sent_idx, sentence in enumerate(sentences):
                    sent_start = sentence["start"]
                    sent_end = sentence["end"]
                    
                    # Check if entity position overlaps with sentence
                    if (entity_start >= sent_start and entity_start < sent_end) or \
                       (entity_end > sent_start and entity_end <= sent_end) or \
                       (entity_start <= sent_start and entity_end >= sent_end):
                        entity_positions[sent_idx].append((entity, pos_idx))
                        
        return entity_positions
    
    def _extract_relationships_with_window(self, entities: List[Entity],
                                         entity_positions: Dict[int, List[Tuple[Entity, int]]],
                                         window_size: int,
                                         sentences: List[Dict[str, Any]],
                                         metadata: Optional[Dict[str, Any]] = None) -> List[Relationship]:
        """
        Extract relationships based on entity co-occurrence within a window of sentences.
        
        Args:
            entities: List of all entities
            entity_positions: Dictionary mapping sentence indices to entity positions
            window_size: Size of the co-occurrence window in sentences
            sentences: List of sentence dictionaries
            metadata: Optional metadata
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Get the relationship type for this window size
        relationship_type = self.relationship_types.get(window_size, "related")
        
        # For each sentence
        for sent_idx in range(len(sentences)):
            # Get entities in the current window
            window_entities = set()
            for window_sent_idx in range(max(0, sent_idx - window_size + 1), sent_idx + 1):
                for entity, _ in entity_positions.get(window_sent_idx, []):
                    window_entities.add(entity)
                    
            # Create relationships between all pairs of entities in the window
            window_entities_list = list(window_entities)
            for i, entity1 in enumerate(window_entities_list):
                for entity2 in window_entities_list[i+1:]:
                    # Skip self-relationships
                    if entity1.id == entity2.id:
                        continue
                        
                    # Calculate confidence based on window size
                    confidence = 1.0 - ((window_size - 1) * 0.2)
                    confidence = max(self.min_confidence, confidence)
                    
                    # Calculate weight based on proximity
                    weight = 1.0 / window_size
                    
                    # Create the relationship
                    relationship = Relationship(
                        source_id=entity1.id,
                        target_id=entity2.id,
                        relationship_type=relationship_type,
                        confidence=confidence,
                        weight=weight,
                        bidirectional=self.bidirectional,
                        extraction_method="co-occurrence"
                    )
                    
                    # Add context from the sentence
                    relationship.context = sentences[sent_idx]["text"]
                    
                    # Add position information
                    relationship.add_position(
                        sentences[sent_idx]["start"],
                        sentences[sent_idx]["end"]
                    )
                    
                    # Add metadata
                    if metadata:
                        relationship.metadata.update(metadata)
                        
                    # Add window information to metadata
                    relationship.metadata["window_size"] = window_size
                    relationship.metadata["sentence_index"] = sent_idx
                    
                    relationships.append(relationship)
                    
        return relationships
    
    def extract_section_relationships(self, text: str, entities: List[Entity],
                                    sections: List[Dict[str, Any]],
                                    metadata: Optional[Dict[str, Any]] = None) -> List[Relationship]:
        """
        Extract relationships based on entity co-occurrence within document sections.
        
        Args:
            text: The document text
            entities: List of entities
            sections: List of section dictionaries with start, end, and name
            metadata: Optional metadata
            
        Returns:
            List of extracted relationships
        """
        if not self.use_sections or not sections:
            return []
            
        relationships = []
        
        # Map entities to sections
        section_entities = defaultdict(set)
        for entity in entities:
            for position in entity.positions:
                entity_start = position["start"]
                entity_end = position["end"]
                
                # Find which section(s) this entity belongs to
                for section_idx, section in enumerate(sections):
                    section_start = section["start"]
                    section_end = section["end"]
                    
                    # Check if entity position overlaps with section
                    if (entity_start >= section_start and entity_start < section_end) or \
                       (entity_end > section_start and entity_end <= section_end) or \
                       (entity_start <= section_start and entity_end >= section_end):
                        section_entities[section_idx].add(entity)
                        
        # Create relationships between entities in the same section
        for section_idx, section_entity_set in section_entities.items():
            section = sections[section_idx]
            section_name = section.get("name", f"Section {section_idx}")
            
            # Create relationships between all pairs of entities in the section
            section_entity_list = list(section_entity_set)
            for i, entity1 in enumerate(section_entity_list):
                for entity2 in section_entity_list[i+1:]:
                    # Skip self-relationships
                    if entity1.id == entity2.id:
                        continue
                        
                    # Create the relationship
                    relationship = Relationship(
                        source_id=entity1.id,
                        target_id=entity2.id,
                        relationship_type="section_related",
                        confidence=0.7,
                        weight=0.5,
                        bidirectional=self.bidirectional,
                        extraction_method="section_co-occurrence"
                    )
                    
                    # Add context from the section name
                    relationship.context = f"Co-occurrence in section: {section_name}"
                    
                    # Add position information
                    relationship.add_position(section["start"], section["end"])
                    
                    # Add metadata
                    if metadata:
                        relationship.metadata.update(metadata)
                        
                    # Add section information to metadata
                    relationship.metadata["section_name"] = section_name
                    relationship.metadata["section_index"] = section_idx
                    
                    relationships.append(relationship)
                    
        return relationships
    
    def extract_proximity_relationships(self, text: str, entities: List[Entity],
                                      metadata: Optional[Dict[str, Any]] = None) -> List[Relationship]:
        """
        Extract relationships based on character proximity between entities.
        
        Args:
            text: The document text
            entities: List of entities
            metadata: Optional metadata
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Sort entities by position
        entity_positions = []
        for entity in entities:
            for pos_idx, position in enumerate(entity.positions):
                entity_positions.append((entity, position["start"], position["end"], pos_idx))
                
        # Sort by start position
        entity_positions.sort(key=lambda x: x[1])
        
        # Create relationships between nearby entities
        for i, (entity1, start1, end1, pos_idx1) in enumerate(entity_positions):
            for j in range(i+1, len(entity_positions)):
                entity2, start2, end2, pos_idx2 = entity_positions[j]
                
                # Skip self-relationships
                if entity1.id == entity2.id:
                    continue
                    
                # Calculate distance between entities
                distance = start2 - end1
                
                # Skip if distance exceeds maximum
                if distance > self.max_distance:
                    break  # Since entities are sorted, all further entities will be too far
                    
                # Calculate confidence and weight based on distance
                confidence = 1.0 - (distance / self.max_distance)
                weight = 1.0 - (distance / self.max_distance)
                
                # Determine relationship type based on distance
                if distance <= 50:
                    rel_type = "very_close"
                elif distance <= 100:
                    rel_type = "close"
                elif distance <= 200:
                    rel_type = "nearby"
                else:
                    rel_type = "distant"
                
                # Create the relationship
                relationship = Relationship(
                    source_id=entity1.id,
                    target_id=entity2.id,
                    relationship_type=rel_type,
                    confidence=confidence,
                    weight=weight,
                    bidirectional=self.bidirectional,
                    extraction_method="proximity"
                )
                
                # Add context from the text between entities
                context_start = max(0, end1 - 10)
                context_end = min(len(text), start2 + 10)
                relationship.context = text[context_start:context_end]
                
                # Add position information
                relationship.add_position(end1, start2)
                
                # Add metadata
                if metadata:
                    relationship.metadata.update(metadata)
                    
                # Add distance information to metadata
                relationship.metadata["distance"] = distance
                
                relationships.append(relationship)
                
        return relationships
