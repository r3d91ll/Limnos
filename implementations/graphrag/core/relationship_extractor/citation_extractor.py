"""
Citation Relationship Extractor for GraphRAG

This module implements relationship extraction based on citations and references
in academic documents, identifying connections between entities through citation patterns.
"""

import re
from typing import Dict, List, Optional, Any, Set, Tuple
import logging
from collections import defaultdict

from ..models.entity import Entity
from ..models.relationship import Relationship
from .relationship_extractor import RelationshipExtractor


class CitationRelationshipExtractor(RelationshipExtractor):
    """
    Relationship extractor implementation based on citations and references.
    
    This extractor identifies explicit relationships between entities based on
    citation patterns in academic documents. It connects entities that are mentioned
    in proximity to citations, as well as linking citations to their corresponding
    references.
    """
    
    # Regular expressions for citation patterns
    CITATION_PATTERNS = {
        "numeric": r'\[(\d+(?:,\s*\d+)*)\]',
        "author_year": r'\(\s*([A-Za-z]+\s+et\s+al\.\s*,\s*\d{4}[a-z]?)\s*\)'
    }
    
    # Regular expression for reference entries
    REFERENCE_PATTERN = r'(?:^|\n)(\[\d+\])\s+(.+?)(?=\n\[\d+\]|\n\n|$)'
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the citation relationship extractor.
        
        Args:
            config: Configuration dictionary with the following options:
                - window_size: Size of text window around citations (default: 200)
                - min_confidence: Minimum confidence threshold (default: 0.7)
                - extract_citation_reference: Whether to extract citation-reference relationships (default: True)
                - extract_entity_citation: Whether to extract entity-citation relationships (default: True)
                - bidirectional: Whether relationships are bidirectional (default: False)
        """
        super().__init__(config)
        
        # Get configuration options
        self.window_size = self.config.get("window_size", 200)
        self.min_confidence = self.config.get("min_confidence", 0.7)
        self.extract_citation_reference = self.config.get("extract_citation_reference", True)
        self.extract_entity_citation = self.config.get("extract_entity_citation", True)
        self.bidirectional = self.config.get("bidirectional", False)
        
        # Compile regular expressions
        self.citation_regex = {
            pattern_name: re.compile(pattern)
            for pattern_name, pattern in self.CITATION_PATTERNS.items()
        }
        self.reference_regex = re.compile(self.REFERENCE_PATTERN, re.DOTALL)
    
    def extract_relationships(self, text: str, entities: List[Entity], 
                             metadata: Optional[Dict[str, Any]] = None) -> List[Relationship]:
        """
        Extract citation-based relationships between entities from the given text.
        
        Args:
            text: The text to extract relationships from
            entities: List of entities to find relationships between
            metadata: Optional metadata about the text
            
        Returns:
            List of extracted relationships
        """
        all_relationships = []
        
        # Extract citation entities
        citation_entities = [e for e in entities if e.entity_type == "citation"]
        reference_entities = [e for e in entities if e.entity_type == "reference"]
        
        # Extract citation-reference relationships
        if self.extract_citation_reference and citation_entities and reference_entities:
            citation_reference_relationships = self._extract_citation_reference_relationships(
                citation_entities, reference_entities, metadata
            )
            all_relationships.extend(citation_reference_relationships)
            
        # Extract entity-citation relationships
        if self.extract_entity_citation and citation_entities:
            entity_citation_relationships = self._extract_entity_citation_relationships(
                text, entities, citation_entities, metadata
            )
            all_relationships.extend(entity_citation_relationships)
            
        return self.deduplicate_relationships(all_relationships)
    
    def _extract_citation_reference_relationships(self, citation_entities: List[Entity],
                                               reference_entities: List[Entity],
                                               metadata: Optional[Dict[str, Any]] = None) -> List[Relationship]:
        """
        Extract relationships between citations and their corresponding references.
        
        Args:
            citation_entities: List of citation entities
            reference_entities: List of reference entities
            metadata: Optional metadata
            
        Returns:
            List of citation-reference relationships
        """
        relationships = []
        
        # Create a mapping of reference IDs to reference entities
        reference_map = {}
        for ref_entity in reference_entities:
            # Extract reference ID from canonical name (e.g., "Reference [1]")
            ref_id_match = re.search(r'\[(\d+)\]', ref_entity.canonical_name)
            if ref_id_match:
                ref_id = ref_id_match.group(1)
                reference_map[ref_id] = ref_entity
                
        # For each citation, find the corresponding reference(s)
        for citation_entity in citation_entities:
            # Get citation IDs from metadata
            citation_ids = citation_entity.metadata.get("citation_ids", [])
            
            for citation_id in citation_ids:
                # For numeric citations, look up the reference directly
                if citation_id.isdigit() and citation_id in reference_map:
                    reference_entity = reference_map[citation_id]
                    
                    # Create the relationship
                    relationship = Relationship(
                        source_id=citation_entity.id,
                        target_id=reference_entity.id,
                        relationship_type="cites",
                        confidence=0.95,
                        weight=1.0,
                        bidirectional=self.bidirectional,
                        extraction_method="citation_reference"
                    )
                    
                    # Add context
                    relationship.context = f"Citation {citation_id} refers to reference {citation_id}"
                    
                    # Add metadata
                    if metadata:
                        relationship.metadata.update(metadata)
                        
                    # Add citation information to metadata
                    relationship.metadata["citation_id"] = citation_id
                    
                    relationships.append(relationship)
                    
        return relationships
    
    def _extract_entity_citation_relationships(self, text: str, entities: List[Entity],
                                            citation_entities: List[Entity],
                                            metadata: Optional[Dict[str, Any]] = None) -> List[Relationship]:
        """
        Extract relationships between entities and citations they appear near.
        
        Args:
            text: The document text
            entities: List of all entities
            citation_entities: List of citation entities
            metadata: Optional metadata
            
        Returns:
            List of entity-citation relationships
        """
        relationships = []
        
        # Filter out citation and reference entities
        content_entities = [e for e in entities 
                          if e.entity_type not in ["citation", "reference"]]
        
        # For each citation entity
        for citation_entity in citation_entities:
            # Get the citation position
            if not citation_entity.positions:
                continue
                
            citation_pos = citation_entity.positions[0]
            citation_start = citation_pos["start"]
            citation_end = citation_pos["end"]
            
            # Define the window around the citation
            window_start = max(0, citation_start - self.window_size)
            window_end = min(len(text), citation_end + self.window_size)
            
            # Find entities within this window
            for entity in content_entities:
                # Skip if entity has no positions
                if not entity.positions:
                    continue
                    
                # Check if any position of the entity is within the window
                entity_in_window = False
                for pos in entity.positions:
                    entity_start = pos["start"]
                    entity_end = pos["end"]
                    
                    # Check if entity overlaps with the window
                    if (entity_start >= window_start and entity_start < window_end) or \
                       (entity_end > window_start and entity_end <= window_end) or \
                       (entity_start <= window_start and entity_end >= window_end):
                        entity_in_window = True
                        break
                        
                if not entity_in_window:
                    continue
                    
                # Calculate confidence based on proximity
                distance = min(
                    abs(citation_start - entity_start),
                    abs(citation_end - entity_start),
                    abs(citation_start - entity_end),
                    abs(citation_end - entity_end)
                )
                
                confidence = 1.0 - (distance / (2 * self.window_size))
                confidence = max(self.min_confidence, confidence)
                
                # Create the relationship
                relationship = Relationship(
                    source_id=entity.id,
                    target_id=citation_entity.id,
                    relationship_type="cited_with",
                    confidence=confidence,
                    weight=confidence,
                    bidirectional=self.bidirectional,
                    extraction_method="entity_citation"
                )
                
                # Add context from the window
                context_start = max(window_start, min(entity_start, citation_start) - 50)
                context_end = min(window_end, max(entity_end, citation_end) + 50)
                relationship.context = text[context_start:context_end]
                
                # Add position information
                relationship.add_position(
                    min(entity_start, citation_start),
                    max(entity_end, citation_end)
                )
                
                # Add metadata
                if metadata:
                    relationship.metadata.update(metadata)
                    
                # Add citation information to metadata
                relationship.metadata["citation_text"] = citation_entity.text
                relationship.metadata["distance"] = distance
                
                relationships.append(relationship)
                
        return relationships
    
    def extract_citation_network(self, entities: List[Entity], 
                               metadata: Optional[Dict[str, Any]] = None) -> List[Relationship]:
        """
        Extract a citation network from the document's citations and references.
        
        Args:
            entities: List of entities (should include citations and references)
            metadata: Optional metadata
            
        Returns:
            List of relationships forming a citation network
        """
        # Extract citation and reference entities
        citation_entities = [e for e in entities if e.entity_type == "citation"]
        reference_entities = [e for e in entities if e.entity_type == "reference"]
        
        # Extract citation-reference relationships
        citation_reference_relationships = self._extract_citation_reference_relationships(
            citation_entities, reference_entities, metadata
        )
        
        # Create a mapping of reference IDs to reference entities
        reference_map = {}
        for ref_entity in reference_entities:
            ref_id_match = re.search(r'\[(\d+)\]', ref_entity.canonical_name)
            if ref_id_match:
                ref_id = ref_id_match.group(1)
                reference_map[ref_id] = ref_entity
                
        # Extract relationships between references (based on citation patterns)
        reference_relationships = []
        
        # Group citations by sentence or paragraph to identify co-citation
        citation_groups = defaultdict(list)
        for citation_entity in citation_entities:
            # Use the first position as a proxy for the sentence/paragraph
            if citation_entity.positions:
                pos = citation_entity.positions[0]
                # Use a rough estimate of sentence/paragraph boundaries
                group_key = pos["start"] // 500  # Group citations within ~500 chars
                citation_groups[group_key].append(citation_entity)
                
        # Create relationships between co-cited references
        for group_key, group_citations in citation_groups.items():
            # Skip groups with only one citation
            if len(group_citations) < 2:
                continue
                
            # Create relationships between all pairs of citations in the group
            for i, citation1 in enumerate(group_citations):
                for citation2 in group_citations[i+1:]:
                    # Get citation IDs
                    citation1_ids = citation1.metadata.get("citation_ids", [])
                    citation2_ids = citation2.metadata.get("citation_ids", [])
                    
                    # Create relationships between the corresponding references
                    for cid1 in citation1_ids:
                        for cid2 in citation2_ids:
                            # Skip self-citations
                            if cid1 == cid2:
                                continue
                                
                            # Look up the reference entities
                            if cid1.isdigit() and cid2.isdigit() and \
                               cid1 in reference_map and cid2 in reference_map:
                                ref1 = reference_map[cid1]
                                ref2 = reference_map[cid2]
                                
                                # Create the relationship
                                relationship = Relationship(
                                    source_id=ref1.id,
                                    target_id=ref2.id,
                                    relationship_type="co_cited_with",
                                    confidence=0.8,
                                    weight=0.8,
                                    bidirectional=True,
                                    extraction_method="co_citation"
                                )
                                
                                # Add context
                                relationship.context = f"References {cid1} and {cid2} are co-cited"
                                
                                # Add metadata
                                if metadata:
                                    relationship.metadata.update(metadata)
                                    
                                # Add citation information to metadata
                                relationship.metadata["citation_group"] = group_key
                                
                                reference_relationships.append(relationship)
                                
        # Combine all relationships
        all_relationships = citation_reference_relationships + reference_relationships
        return self.deduplicate_relationships(all_relationships)
