"""
Entity Linker for GraphRAG

This module provides functionality for entity linking and disambiguation,
connecting extracted entities to canonical representations in the knowledge graph.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
import logging
from collections import defaultdict
import networkx as nx
from difflib import SequenceMatcher
import re

from ..models.entity import Entity


class EntityLinker:
    """
    Entity linker for connecting extracted entities to canonical representations.
    
    The EntityLinker is responsible for:
    1. Disambiguating entities (resolving multiple mentions to the same entity)
    2. Linking entities to existing entities in the knowledge graph
    3. Normalizing entity names and attributes
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the entity linker.
        
        Args:
            config: Configuration dictionary with the following options:
                - similarity_threshold: Minimum similarity for entity matching (default: 0.85)
                - case_sensitive: Whether matching should be case-sensitive (default: False)
                - use_aliases: Whether to use entity aliases for matching (default: True)
                - max_candidates: Maximum number of candidates to consider (default: 5)
                - entity_types_strict: Whether entity types must match exactly (default: True)
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configuration options
        self.similarity_threshold = self.config.get("similarity_threshold", 0.85)
        self.case_sensitive = self.config.get("case_sensitive", False)
        self.use_aliases = self.config.get("use_aliases", True)
        self.max_candidates = self.config.get("max_candidates", 5)
        self.entity_types_strict = self.config.get("entity_types_strict", True)
        
        # Entity cache for quick lookup
        self.entity_cache = {}
        self.fingerprint_cache = {}
        
    def link_entities(self, new_entities: List[Entity], 
                     existing_entities: List[Entity]) -> List[Entity]:
        """
        Link new entities to existing entities.
        
        Args:
            new_entities: List of newly extracted entities
            existing_entities: List of existing entities in the knowledge graph
            
        Returns:
            List of linked entities (merged where appropriate)
        """
        self.logger.info(f"Linking {len(new_entities)} new entities to {len(existing_entities)} existing entities")
        
        # Build cache of existing entities if not already built
        if not self.entity_cache:
            self._build_entity_cache(existing_entities)
            
        # Process each new entity
        linked_entities = []
        for new_entity in new_entities:
            # Find matching existing entities
            matches = self._find_matching_entities(new_entity)
            
            if not matches:
                # No match found, add as a new entity
                linked_entities.append(new_entity)
                continue
                
            # Get the best match
            best_match, similarity = matches[0]
            
            # Merge the new entity into the best match
            merged_entity = best_match.merge(new_entity)
            linked_entities.append(merged_entity)
            
        self.logger.info(f"Linked entities: {len(linked_entities)}")
        return linked_entities
    
    def _build_entity_cache(self, entities: List[Entity]) -> None:
        """
        Build caches for efficient entity lookup.
        
        Args:
            entities: List of entities to cache
        """
        self.entity_cache = defaultdict(list)
        self.fingerprint_cache = {}
        
        for entity in entities:
            # Cache by entity type
            self.entity_cache[entity.entity_type].append(entity)
            
            # Cache by fingerprint
            self.fingerprint_cache[entity.fingerprint] = entity
            
            # Cache by canonical name
            key = entity.canonical_name.lower() if not self.case_sensitive else entity.canonical_name
            if key not in self.entity_cache:
                self.entity_cache[key] = []
            self.entity_cache[key].append(entity)
            
            # Cache by aliases if enabled
            if self.use_aliases:
                for alias in entity.aliases:
                    alias_key = alias.lower() if not self.case_sensitive else alias
                    if alias_key not in self.entity_cache:
                        self.entity_cache[alias_key] = []
                    self.entity_cache[alias_key].append(entity)
    
    def _find_matching_entities(self, entity: Entity) -> List[Tuple[Entity, float]]:
        """
        Find existing entities that match the given entity.
        
        Args:
            entity: Entity to find matches for
            
        Returns:
            List of (matching_entity, similarity_score) tuples, sorted by similarity
        """
        # Check for exact fingerprint match first
        if entity.fingerprint in self.fingerprint_cache:
            matching_entity = self.fingerprint_cache[entity.fingerprint]
            return [(matching_entity, 1.0)]
            
        # Get candidate entities of the same type
        candidates = []
        if self.entity_types_strict:
            # Only consider entities of the same type
            candidates = self.entity_cache.get(entity.entity_type, [])
        else:
            # Consider all entities
            for entity_list in self.entity_cache.values():
                candidates.extend(entity_list)
                
        # If we have a canonical name match, prioritize those candidates
        name_key = entity.canonical_name.lower() if not self.case_sensitive else entity.canonical_name
        name_matches = self.entity_cache.get(name_key, [])
        
        # Combine and deduplicate candidates
        all_candidates = list(set(name_matches + candidates))
        
        # Calculate similarity for each candidate
        matches = []
        for candidate in all_candidates:
            similarity = self._calculate_similarity(entity, candidate)
            if similarity >= self.similarity_threshold:
                matches.append((candidate, similarity))
                
        # Sort by similarity (descending) and limit to max_candidates
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:self.max_candidates]
    
    def _calculate_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """
        Calculate similarity between two entities.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            Similarity score between 0 and 1
        """
        # If entity types don't match and we're strict about types, return 0
        if self.entity_types_strict and entity1.entity_type != entity2.entity_type:
            return 0.0
            
        # Start with name similarity
        name1 = entity1.canonical_name.lower() if not self.case_sensitive else entity1.canonical_name
        name2 = entity2.canonical_name.lower() if not self.case_sensitive else entity2.canonical_name
        
        # Use SequenceMatcher for string similarity
        name_similarity = SequenceMatcher(None, name1, name2).ratio()
        
        # Check aliases if enabled
        if self.use_aliases:
            # Get all aliases
            aliases1 = set(a.lower() if not self.case_sensitive else a for a in entity1.aliases)
            aliases2 = set(a.lower() if not self.case_sensitive else a for a in entity2.aliases)
            
            # Calculate maximum alias similarity
            max_alias_similarity = 0.0
            for alias1 in aliases1:
                for alias2 in aliases2:
                    alias_similarity = SequenceMatcher(None, alias1, alias2).ratio()
                    max_alias_similarity = max(max_alias_similarity, alias_similarity)
                    
            # Use the higher of name similarity or alias similarity
            return max(name_similarity, max_alias_similarity)
            
        return name_similarity
    
    def disambiguate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Disambiguate a list of entities, merging duplicates.
        
        Args:
            entities: List of entities to disambiguate
            
        Returns:
            List of disambiguated entities
        """
        self.logger.info(f"Disambiguating {len(entities)} entities")
        
        # Group entities by fingerprint
        fingerprint_groups = defaultdict(list)
        for entity in entities:
            fingerprint_groups[entity.fingerprint].append(entity)
            
        # Merge entities within each group
        disambiguated = []
        for fingerprint, group in fingerprint_groups.items():
            if len(group) == 1:
                # Only one entity with this fingerprint
                disambiguated.append(group[0])
            else:
                # Merge all entities in the group
                merged = group[0]
                for entity in group[1:]:
                    merged = merged.merge(entity)
                disambiguated.append(merged)
                
        self.logger.info(f"Disambiguated to {len(disambiguated)} entities")
        return disambiguated
    
    def normalize_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Normalize entity attributes for consistency.
        
        Args:
            entities: List of entities to normalize
            
        Returns:
            List of normalized entities
        """
        normalized = []
        
        for entity in entities:
            # Create a copy to avoid modifying the original
            normalized_entity = Entity(
                text=entity.text,
                entity_type=entity.entity_type,
                id=entity.id,
                canonical_name=entity.canonical_name,
                confidence=entity.confidence,
                source_document_id=entity.source_document_id,
                positions=entity.positions.copy(),
                metadata=entity.metadata.copy(),
                aliases=entity.aliases.copy()
            )
            
            # Normalize canonical name (remove extra whitespace, etc.)
            normalized_entity.canonical_name = self._normalize_text(normalized_entity.canonical_name)
            
            # Normalize text
            normalized_entity.text = self._normalize_text(normalized_entity.text)
            
            # Normalize aliases
            normalized_aliases = set()
            for alias in normalized_entity.aliases:
                normalized_aliases.add(self._normalize_text(alias))
            normalized_entity.aliases = normalized_aliases
            
            normalized.append(normalized_entity)
            
        return normalized
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing extra whitespace, etc.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
            
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common punctuation from the beginning and end
        normalized = re.sub(r'^[\s.,;:!?"\'-]+|[\s.,;:!?"\'-]+$', '', normalized)
        
        return normalized
    
    def build_entity_graph(self, entities: List[Entity]) -> nx.Graph:
        """
        Build a graph of entity relationships based on co-occurrence.
        
        Args:
            entities: List of entities to build the graph from
            
        Returns:
            NetworkX graph of entity relationships
        """
        # Create a new undirected graph
        graph = nx.Graph()
        
        # Add entities as nodes
        for entity in entities:
            graph.add_node(entity.id, 
                          entity_type=entity.entity_type,
                          label=entity.canonical_name,
                          entity=entity)
            
        # Create edges based on co-occurrence in the same document
        doc_entities = defaultdict(list)
        for entity in entities:
            if entity.source_document_id:
                doc_entities[entity.source_document_id].append(entity)
                
        # Connect entities that appear in the same document
        for doc_id, doc_entity_list in doc_entities.items():
            for i, entity1 in enumerate(doc_entity_list):
                for entity2 in doc_entity_list[i+1:]:
                    # Skip self-connections
                    if entity1.id == entity2.id:
                        continue
                        
                    # Add or update edge
                    if graph.has_edge(entity1.id, entity2.id):
                        # Increment weight if edge already exists
                        graph[entity1.id][entity2.id]['weight'] += 1
                    else:
                        # Create new edge
                        graph.add_edge(entity1.id, entity2.id, 
                                      edge_type="co-occurrence",
                                      weight=1)
                        
        return graph
