"""
Relationship Extractor for PathRAG

This module provides relationship extraction capabilities for the PathRAG implementation.
It identifies relationships between entities that will serve as edges in the path graph.
"""

import re
from typing import List, Dict, Any, Tuple, Set, Optional
import logging
import networkx as nx
import spacy
from spacy.tokens import Doc

# Configure logging
logger = logging.getLogger(__name__)

class RelationshipExtractor:
    """
    Extracts relationships between entities to serve as edges in the PathRAG graph.
    
    This extractor identifies semantic relationships that connect entities,
    forming the edges in the path-based retrieval graph.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the relationship extractor with configuration.
        
        Args:
            config: Configuration dictionary with options for relationship extraction
        """
        self.config = config or {}
        self.nlp = None
        self.initialized = False
        
        # Window size for co-occurrence relationships
        self.co_occurrence_window = self.config.get("co_occurrence_window", 50)
        
        # Minimum frequency for relationship to be considered valid
        self.min_relationship_frequency = self.config.get("min_relationship_frequency", 1)
        
        # Whether to extract dependency-based relationships
        self.use_dependency_parsing = self.config.get("use_dependency_parsing", True)
        
        # Standard relationship types
        self.relationship_types = {
            "CO_OCCURS_WITH": "Entities appear near each other",
            "HAS_ATTRIBUTE": "Entity has a specific attribute",
            "PART_OF": "Entity is part of another entity",
            "BELONGS_TO": "Entity belongs to another entity",
            "LOCATED_IN": "Entity is located in another entity",
            "CREATED_BY": "Entity was created by another entity",
            "USED_FOR": "Entity is used for a purpose",
            "RELATED_TO": "General relationship between entities",
            "INSTANCE_OF": "Entity is an instance of a class",
            "CAUSES": "Entity causes another entity",
            "PRECEDES": "Entity precedes another entity in sequence",
            "FOLLOWED_BY": "Entity is followed by another entity",
            "CONTRADICTS": "Entity contradicts another entity"
        }
        
        # Add custom relationship types from config
        if "custom_relationship_types" in self.config:
            self.relationship_types.update(self.config["custom_relationship_types"])
    
    def initialize(self) -> None:
        """Initialize the NLP pipeline for relationship extraction."""
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
        
        self.initialized = True
    
    def extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities in the given text.
        
        Args:
            text: The text content to extract relationships from
            entities: List of entities previously extracted from the text
            
        Returns:
            List of relationship dictionaries with metadata
        """
        if not self.initialized:
            self.initialize()
        
        # Process the text with spaCy
        doc = self.nlp(text)
        
        # Create a mapping of entity spans to entity dictionaries
        entity_spans = {}
        for entity in entities:
            # Create a tuple of (start_char, end_char) for lookup
            span_key = (entity["start"], entity["end"])
            entity_spans[span_key] = entity
        
        # Extract relationships
        relationships = []
        
        # 1. Extract co-occurrence relationships
        co_occurrences = self._extract_co_occurrences(doc, entity_spans)
        relationships.extend(co_occurrences)
        
        # 2. Extract dependency-based relationships if enabled
        if self.use_dependency_parsing:
            dependency_relations = self._extract_dependency_relations(doc, entity_spans)
            relationships.extend(dependency_relations)
        
        # 3. Extract verb-mediated relationships
        verb_relations = self._extract_verb_relations(doc, entity_spans)
        relationships.extend(verb_relations)
        
        # 4. Extract preposition-based relationships
        preposition_relations = self._extract_preposition_relations(doc, entity_spans)
        relationships.extend(preposition_relations)
        
        # Add unique IDs to relationships
        for i, rel in enumerate(relationships):
            src_id = rel["source"]["id"]
            tgt_id = rel["target"]["id"]
            rel_type = rel["type"]
            rel["id"] = f"rel_{i}_{src_id}_{rel_type}_{tgt_id}"
        
        return relationships
    
    def _extract_co_occurrences(self, doc: Doc, entity_spans: Dict[Tuple[int, int], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract co-occurrence relationships between entities.
        
        Args:
            doc: spaCy Doc object
            entity_spans: Mapping of (start, end) spans to entity dictionaries
            
        Returns:
            List of co-occurrence relationship dictionaries
        """
        co_occurrences = []
        entity_items = list(entity_spans.items())
        
        for i, ((start1, end1), entity1) in enumerate(entity_items):
            for ((start2, end2), entity2) in entity_items[i+1:]:
                # Check if entities are within the co-occurrence window
                if abs(start1 - start2) <= self.co_occurrence_window:
                    # Create co-occurrence relationship
                    relationship = {
                        "source": entity1,
                        "target": entity2,
                        "type": "CO_OCCURS_WITH",
                        "confidence": 0.7,  # Base confidence for co-occurrence
                        "metadata": {
                            "distance": abs(start1 - start2)
                        }
                    }
                    
                    # Adjust confidence based on distance (closer = higher confidence)
                    relationship["confidence"] = max(0.5, 1.0 - (abs(start1 - start2) / self.co_occurrence_window))
                    
                    co_occurrences.append(relationship)
        
        return co_occurrences
    
    def _extract_dependency_relations(self, doc: Doc, entity_spans: Dict[Tuple[int, int], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships based on dependency parsing.
        
        Args:
            doc: spaCy Doc object
            entity_spans: Mapping of (start, end) spans to entity dictionaries
            
        Returns:
            List of dependency-based relationship dictionaries
        """
        dependency_relations = []
        
        # Create a mapping of token indices to entity dictionaries
        token_to_entity = {}
        for (start, end), entity in entity_spans.items():
            # Find all tokens that are part of this entity
            for token in doc:
                if start <= token.idx < end:
                    token_to_entity[token.i] = entity
        
        # Analyze dependency structure
        for token in doc:
            # Skip tokens that are not part of any entity
            if token.i not in token_to_entity:
                continue
                
            source_entity = token_to_entity[token.i]
            
            # Check dependencies where this token is the head
            for child in token.children:
                # Skip if child is not part of any entity
                if child.i not in token_to_entity:
                    continue
                    
                target_entity = token_to_entity[child.i]
                
                # Skip self-relations
                if source_entity["id"] == target_entity["id"]:
                    continue
                
                # Determine relationship type based on dependency label
                rel_type = self._map_dependency_to_relationship(child.dep_)
                
                if rel_type:
                    relationship = {
                        "source": source_entity,
                        "target": target_entity,
                        "type": rel_type,
                        "confidence": 0.8,  # Higher confidence for dependency-based relations
                        "metadata": {
                            "dependency": child.dep_,
                            "text": doc[min(token.i, child.i):max(token.i, child.i)+1].text
                        }
                    }
                    dependency_relations.append(relationship)
        
        return dependency_relations
    
    def _map_dependency_to_relationship(self, dep_label: str) -> Optional[str]:
        """
        Map dependency labels to relationship types.
        
        Args:
            dep_label: Dependency label from spaCy
            
        Returns:
            Relationship type string or None if no mapping
        """
        # Mapping of common dependency labels to relationship types
        dep_to_rel = {
            "nsubj": "RELATED_TO",  # Nominal subject
            "nsubjpass": "RELATED_TO",  # Passive nominal subject
            "dobj": "RELATED_TO",  # Direct object
            "pobj": "RELATED_TO",  # Object of preposition
            "amod": "HAS_ATTRIBUTE",  # Adjectival modifier
            "compound": "PART_OF",  # Compound word
            "poss": "BELONGS_TO",  # Possession modifier
            "prep": "RELATED_TO",  # Prepositional modifier
            "attr": "HAS_ATTRIBUTE",  # Attribute
            "appos": "INSTANCE_OF",  # Appositional modifier
            "acl": "RELATED_TO",  # Adjectival clause
            "agent": "CREATED_BY",  # Agent
            "conj": "RELATED_TO",  # Conjunct
        }
        
        return dep_to_rel.get(dep_label)
    
    def _extract_verb_relations(self, doc: Doc, entity_spans: Dict[Tuple[int, int], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships mediated by verbs between entities.
        
        Args:
            doc: spaCy Doc object
            entity_spans: Mapping of (start, end) spans to entity dictionaries
            
        Returns:
            List of verb-mediated relationship dictionaries
        """
        verb_relations = []
        
        # Find verbs in the document
        verbs = [token for token in doc if token.pos_ == "VERB"]
        
        for verb in verbs:
            # Find subject and object connected to this verb
            subjects = []
            objects = []
            
            for child in verb.children:
                # Subject relationships
                if child.dep_ in ("nsubj", "nsubjpass"):
                    # Look for entities that include this token
                    for (start, end), entity in entity_spans.items():
                        if start <= child.idx < end:
                            subjects.append(entity)
                
                # Object relationships
                if child.dep_ in ("dobj", "pobj", "attr"):
                    # Look for entities that include this token
                    for (start, end), entity in entity_spans.items():
                        if start <= child.idx < end:
                            objects.append(entity)
            
            # Create relationships between subjects and objects via the verb
            for subject in subjects:
                for obj in objects:
                    # Skip self-relations
                    if subject["id"] == obj["id"]:
                        continue
                    
                    # Determine relation type based on verb semantics
                    rel_type = self._determine_verb_relation_type(verb.lemma_)
                    
                    relationship = {
                        "source": subject,
                        "target": obj,
                        "type": rel_type,
                        "confidence": 0.85,  # High confidence for verb-mediated relations
                        "metadata": {
                            "verb": verb.text,
                            "verb_lemma": verb.lemma_,
                            "text": doc[verb.i-1:verb.i+2].text  # Context around the verb
                        }
                    }
                    verb_relations.append(relationship)
        
        return verb_relations
    
    def _determine_verb_relation_type(self, verb_lemma: str) -> str:
        """
        Determine relationship type based on verb semantics.
        
        Args:
            verb_lemma: Lemmatized verb
            
        Returns:
            Relationship type string
        """
        # Common verb mappings to relationship types
        verb_mappings = {
            "be": "INSTANCE_OF",
            "have": "HAS_ATTRIBUTE",
            "contain": "PART_OF",
            "include": "PART_OF",
            "consist": "PART_OF",
            "belong": "BELONGS_TO",
            "own": "BELONGS_TO",
            "locate": "LOCATED_IN",
            "create": "CREATED_BY",
            "make": "CREATED_BY",
            "produce": "CREATED_BY",
            "use": "USED_FOR",
            "cause": "CAUSES",
            "precede": "PRECEDES",
            "follow": "FOLLOWED_BY",
            "contradict": "CONTRADICTS"
        }
        
        return verb_mappings.get(verb_lemma, "RELATED_TO")
    
    def _extract_preposition_relations(self, doc: Doc, entity_spans: Dict[Tuple[int, int], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships based on prepositional phrases.
        
        Args:
            doc: spaCy Doc object
            entity_spans: Mapping of (start, end) spans to entity dictionaries
            
        Returns:
            List of preposition-based relationship dictionaries
        """
        preposition_relations = []
        
        # Find prepositions in the document
        prepositions = [token for token in doc if token.pos_ == "ADP"]
        
        for prep in prepositions:
            # Find the head of the prepositional phrase
            head = prep.head
            
            # Find the object of the preposition
            pobj = None
            for child in prep.children:
                if child.dep_ == "pobj":
                    pobj = child
                    break
            
            if pobj is None:
                continue
            
            # Find entities corresponding to head and object
            head_entities = []
            obj_entities = []
            
            for (start, end), entity in entity_spans.items():
                # Check if the head is part of this entity
                if start <= head.idx < end:
                    head_entities.append(entity)
                
                # Check if the object is part of this entity
                if pobj and start <= pobj.idx < end:
                    obj_entities.append(entity)
            
            # Create relationships between head entities and object entities
            for head_entity in head_entities:
                for obj_entity in obj_entities:
                    # Skip self-relations
                    if head_entity["id"] == obj_entity["id"]:
                        continue
                    
                    # Determine relation type based on preposition
                    rel_type = self._map_preposition_to_relationship(prep.text)
                    
                    relationship = {
                        "source": head_entity,
                        "target": obj_entity,
                        "type": rel_type,
                        "confidence": 0.8,
                        "metadata": {
                            "preposition": prep.text,
                            "text": doc[prep.i-1:pobj.i+1].text  # Context around the prepositional phrase
                        }
                    }
                    preposition_relations.append(relationship)
        
        return preposition_relations
    
    def _map_preposition_to_relationship(self, preposition: str) -> str:
        """
        Map prepositions to relationship types.
        
        Args:
            preposition: Preposition text
            
        Returns:
            Relationship type string
        """
        # Common preposition mappings to relationship types
        preposition_mappings = {
            "in": "LOCATED_IN",
            "at": "LOCATED_IN",
            "on": "LOCATED_IN",
            "of": "PART_OF",
            "from": "ORIGINATED_FROM",
            "by": "CREATED_BY",
            "with": "RELATED_TO",
            "for": "USED_FOR",
            "to": "RELATED_TO",
            "through": "TRAVERSES",
            "during": "TEMPORAL",
            "before": "PRECEDES",
            "after": "FOLLOWED_BY",
            "about": "RELATED_TO",
            "as": "INSTANCE_OF"
        }
        
        return preposition_mappings.get(preposition.lower(), "RELATED_TO")
    
    def extract_document_relationships(
        self, document: Dict[str, Any], entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships from a document dictionary.
        
        Args:
            document: Document dictionary with 'content' field containing text
            entities: List of entities previously extracted from the document
            
        Returns:
            List of relationship dictionaries with metadata
        """
        if not document or "content" not in document:
            logger.warning("Document missing 'content' field")
            return []
            
        # Extract relationships from document content
        relationships = self.extract_relationships(document["content"], entities)
        
        # Add document reference to each relationship
        for relationship in relationships:
            relationship["document_id"] = document.get("id", "unknown")
            relationship["document_title"] = document.get("title", "")
            
        return relationships
    
    def get_relationship_types(self) -> Dict[str, str]:
        """
        Get all relationship types supported by this extractor.
        
        Returns:
            Dictionary of relationship type strings and their descriptions
        """
        return self.relationship_types.copy()
