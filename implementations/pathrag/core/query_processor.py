"""
Path-based Query Processor for PathRAG (Hermes)

This module provides a query processor that transforms natural language queries
into path-based retrieval operations for the PathRAG implementation.

The query processor, named Hermes (messenger of the gods), analyzes query semantics,
extracts key entities and relationships, and constructs path queries optimized
for retrieval performance.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional, Union, Set
from enum import Enum
import re
import numpy as np

from .entity_extractor import EntityExtractor
from .relationship_extractor import RelationshipExtractor
from .path_constructor import PathConstructor
from .path_structures import Path as RagPath, PathNode, PathEdge  # Rename to avoid conflicts with pathlib.Path
from .path_vector_store import PathVectorStore
from .path_storage_manager import PathStorageManager

# Configure logging
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Enumeration of query types supported by the processor."""
    FACTOID = "factoid"  # Simple factual questions (Who, What, When, Where)
    RELATIONSHIP = "relationship"  # Questions about relationships between entities
    CAUSAL = "causal"  # Questions about causality (Why, How)
    COMPARATIVE = "comparative"  # Comparisons between entities
    EXPLORATORY = "exploratory"  # Open-ended exploration

class QueryProcessor:
    """
    Transforms natural language queries into path-based retrieval operations.
    
    This component serves as the entry point for query processing in the PathRAG
    framework, analyzing query semantics and generating appropriate path-based
    retrieval strategies.
    
    Also known as Hermes in the Limnos mythology-based naming scheme.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the query processor with configuration.
        
        Args:
            config: Configuration dictionary with options for the query processor
        """
        self.config = config or {}
        
        # Initialize entity and relationship extractors
        self.entity_extractor: EntityExtractor = EntityExtractor(self.config.get('entity_extractor_config', {}))
        self.relationship_extractor: RelationshipExtractor = RelationshipExtractor(self.config.get('relationship_extractor_config', {}))
        self.path_constructor: PathConstructor = PathConstructor(self.config.get('path_constructor_config', {}))
        
        # Initialize storage manager
        self.storage_manager: Optional[PathStorageManager] = None
        if self.config.get('initialize_storage', True):
            self.storage_manager = PathStorageManager(self.config.get('storage_manager_config', {}))
        
        # Query classification patterns
        self.query_patterns: Dict[QueryType, List[str]] = {
            QueryType.FACTOID: [
                r"^(who|what|when|where)\s", 
                r"^(name|list|identify|tell me)\s",
                r"(is|are|was|were)\s.+\?"
            ],
            QueryType.RELATIONSHIP: [
                r"(relationship|relation|connection|link)\s+between",
                r"how\s+(is|are|was|were|do|does|did).+related",
                r"how\s+do.*\s+interact",
                r"what\s+(is|are).*\s+between"
            ],
            QueryType.CAUSAL: [
                r"^(why|how)\s",
                r"(cause|effect|impact|influence|result\s+in|lead\s+to)",
                r"what\s+(causes|caused|effects|impacts|influences)",
                r"what\s+is\s+the\s+(cause|effect|impact|influence|result)\s+of"
            ],
            QueryType.COMPARATIVE: [
                r"(compare|comparison|difference|similar|similarly|versus|vs\.)",
                r"(better|worse|stronger|weaker|more|less|faster|slower)\s+than",
                r"what\s+(is|are)\s+the\s+differences?\s+between"
            ],
            QueryType.EXPLORATORY: [
                r"(tell me about|explain|describe|elaborate on|discuss)",
                r"^(what|how)\s+(do|does|is|are)\s+.+\s+(work|function)",
                r"I'm interested in|I want to know about"
            ]
        }
        
        # Maximum number of entities to extract from a query
        self.max_query_entities: int = self.config.get('max_query_entities', 5)
        
        # Maximum number of search paths to generate
        self.max_search_paths: int = self.config.get('max_search_paths', 10)
        
        # Whether to perform vector-based semantic search alongside path search
        self.use_semantic_search: bool = self.config.get('use_semantic_search', True)
        
        # Threshold for entity confidence in queries
        self.entity_confidence_threshold: float = self.config.get('entity_confidence_threshold', 0.3)
    
    def process_query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a natural language query into a structured retrieval plan.
        
        Args:
            query_text: The natural language query text
            
        Returns:
            A dictionary containing the structured query plan and metadata
        """
        # Preprocess the query
        clean_query = self._preprocess_query(query_text)
        
        # Classify the query type
        query_type = self._classify_query(clean_query)
        logger.info(f"Classified query as {query_type.value}: {clean_query}")
        
        # Extract entities from the query
        query_entities = self._extract_query_entities(clean_query)
        
        # Generate search paths based on query type and entities
        search_paths = self._generate_search_paths(clean_query, query_type, query_entities)
        
        # Determine if semantic search should be used
        use_semantic_search = self._should_use_semantic_search(query_type, query_entities)
        
        # Build the final query plan
        query_plan = {
            'original_query': query_text,
            'processed_query': clean_query,
            'query_type': query_type.value,
            'entities': query_entities,
            'search_paths': search_paths,
            'use_semantic_search': use_semantic_search,
            'search_parameters': self._generate_search_parameters(
                clean_query, query_type, query_entities, search_paths
            )
        }
        
        logger.info(f"Generated query plan with {len(search_paths)} search paths and {len(query_entities)} entities")
        return query_plan
    
    def _preprocess_query(self, query_text: str) -> str:
        """
        Preprocess the query text for better processing.
        
        Args:
            query_text: The raw query text
            
        Returns:
            Preprocessed query text
        """
        # Convert to lowercase
        text = query_text.lower()
        
        # Remove punctuation except for question marks
        text = re.sub(r'[^\w\s\?]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure there's a question mark at the end if none exists
        if not text.endswith('?'):
            text += '?'
        
        return text
    
    def _classify_query(self, query_text: str) -> QueryType:
        """
        Classify the query into one of the supported query types.
        
        Args:
            query_text: The preprocessed query text
            
        Returns:
            QueryType enum value
        """
        scores = {qtype: 0 for qtype in QueryType}
        
        # Check patterns for each query type
        for qtype, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_text, re.IGNORECASE):
                    scores[qtype] += 1
        
        # Default to factoid if no patterns match
        max_score = max(scores.values())
        if max_score == 0:
            return QueryType.FACTOID
        
        # Return the query type with the highest score
        for qtype, score in scores.items():
            if score == max_score:
                return qtype
        
        # Fallback to factoid
        return QueryType.FACTOID
    
    def _extract_query_entities(self, query_text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from the query.
        
        Args:
            query_text: The preprocessed query text
            
        Returns:
            List of entity dictionaries
        """
        # Extract entities using the entity extractor
        all_entities = self.entity_extractor.extract_entities(query_text)
        
        # Filter out low-confidence entities
        filtered_entities = [
            entity for entity in all_entities 
            if entity.get('confidence', 0) >= self.entity_confidence_threshold
        ]
        
        # Sort by confidence and position in the query
        sorted_entities = sorted(
            filtered_entities, 
            key=lambda e: (e.get('confidence', 0), e.get('position', 0)), 
            reverse=True
        )
        
        # Limit the number of entities
        return sorted_entities[:self.max_query_entities]
    
    def _generate_search_paths(
        self, 
        query_text: str, 
        query_type: QueryType, 
        query_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate search paths based on the query type and entities.
        
        Args:
            query_text: The preprocessed query text
            query_type: The classified query type
            query_entities: The extracted entities from the query
            
        Returns:
            List of search path dictionaries
        """
        search_paths: List[Dict[str, Any]] = []
        
        # If no entities, return empty list
        if not query_entities:
            return search_paths
        
        # Extract relationships from the query
        relationships = self.relationship_extractor.extract_relationships(query_text, query_entities)
        
        # Create paths based on the query type
        if query_type == QueryType.FACTOID:
            # For factoid queries, create simple paths starting from each entity
            for entity in query_entities:
                path = {
                    'start_entity': entity,
                    'relationships': [],
                    'max_depth': 2,
                    'max_paths': 3
                }
                search_paths.append(path)
                
        elif query_type == QueryType.RELATIONSHIP:
            # For relationship queries, create paths between entity pairs
            if len(query_entities) >= 2:
                for i in range(len(query_entities)):
                    for j in range(i+1, len(query_entities)):
                        path = {
                            'start_entity': query_entities[i],
                            'end_entity': query_entities[j],
                            'direct_relationships': True,
                            'bidirectional': True,
                            'max_depth': 3,
                            'max_paths': 5
                        }
                        search_paths.append(path)
            else:
                # If only one entity, create a path to find related entities
                path = {
                    'start_entity': query_entities[0],
                    'relationship_types': [r.get('type') for r in relationships],
                    'max_depth': 2,
                    'max_paths': 5
                }
                search_paths.append(path)
                
        elif query_type == QueryType.CAUSAL:
            # For causal queries, look for cause-effect relationships
            causal_types = ['causes', 'leads_to', 'results_in', 'affects', 'influences']
            
            for entity in query_entities:
                path = {
                    'start_entity': entity,
                    'relationship_types': causal_types,
                    'max_depth': 3,
                    'max_paths': 5
                }
                search_paths.append(path)
                
        elif query_type == QueryType.COMPARATIVE:
            # For comparative queries, find paths for each entity for later comparison
            for entity in query_entities:
                path = {
                    'start_entity': entity,
                    'max_depth': 2,
                    'max_paths': 3,
                    'include_properties': True
                }
                search_paths.append(path)
                
        elif query_type == QueryType.EXPLORATORY:
            # For exploratory queries, create diverse paths with varying depths
            for entity in query_entities:
                path = {
                    'start_entity': entity,
                    'max_depth': 3,
                    'max_paths': 5,
                    'diverse_paths': True
                }
                search_paths.append(path)
        
        # Limit the number of search paths
        return search_paths[:self.max_search_paths]
    
    def _should_use_semantic_search(
        self, 
        query_type: QueryType, 
        query_entities: List[Dict[str, Any]]
    ) -> bool:
        """
        Determine if semantic search should be used alongside path search.
        
        Args:
            query_type: The classified query type
            query_entities: The extracted entities from the query
            
        Returns:
            Boolean indicating whether to use semantic search
        """
        # If disabled in config, don't use
        if not self.use_semantic_search:
            return False
        
        # Always use for exploratory queries
        if query_type == QueryType.EXPLORATORY:
            return True
        
        # Use for factoid queries with few entities
        if query_type == QueryType.FACTOID and len(query_entities) <= 1:
            return True
        
        # Use for causal queries (often complex)
        if query_type == QueryType.CAUSAL:
            return True
        
        # Default case - use if few entities found
        return len(query_entities) <= 1
    
    def _generate_search_parameters(
        self, 
        query_text: str, 
        query_type: QueryType, 
        query_entities: List[Dict[str, Any]],
        search_paths: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate search parameters for the retrieval system.
        
        Args:
            query_text: The preprocessed query text
            query_type: The classified query type
            query_entities: The extracted entities from the query
            search_paths: The generated search paths
            
        Returns:
            Dictionary of search parameters
        """
        # Base parameters
        params = {
            'entity_texts': [entity.get('text') for entity in query_entities],
            'entity_types': list(set(entity.get('type') for entity in query_entities if entity.get('type'))),
            'max_results': 10,
            'semantic_query': query_text if self._should_use_semantic_search(query_type, query_entities) else None
        }
        
        # Add parameters based on query type
        if query_type == QueryType.RELATIONSHIP:
            params['relationship_types'] = list(set(
                edge.get('type') for path in search_paths 
                for edge in path.get('relationships', [])
            ))
            params['bidirectional'] = True
            
        elif query_type == QueryType.CAUSAL:
            params['relationship_types'] = ['causes', 'leads_to', 'results_in', 'affects', 'influences']
            params['max_results'] = 15  # Increase for causal queries
            
        elif query_type == QueryType.COMPARATIVE:
            params['include_properties'] = True
            params['max_results'] = 20  # Increase for comparative queries
            
        elif query_type == QueryType.EXPLORATORY:
            params['diverse_results'] = True
            params['max_results'] = 25  # Increase for exploratory queries
        
        return params
    
    def execute_query_plan(self, query_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute a query plan to retrieve paths and documents.
        
        Args:
            query_plan: The query plan generated by process_query
            
        Returns:
            List of retrieved paths and associated documents
        """
        if not self.storage_manager:
            raise ValueError("Storage manager not initialized. Cannot execute query.")
        
        search_params = query_plan.get('search_parameters', {})
        
        # Prepare search parameters
        entity_texts = search_params.get('entity_texts', [])
        entity_types = search_params.get('entity_types', [])
        relationship_types = search_params.get('relationship_types', [])
        semantic_query = search_params.get('semantic_query')
        max_results = search_params.get('max_results', 10)
        
        # Execute the advanced search
        if self.storage_manager is None:
            raise ValueError("Storage manager not initialized. Cannot execute query.")
        
        paths = self.storage_manager.advanced_search(
            entity_texts=entity_texts,
            entity_types=entity_types,
            relationship_types=relationship_types,
            query=semantic_query,
            k=max_results
        )
        
        # Prepare the results
        results = []
        
        for path in paths:
            # Get document IDs from path
            document_ids = list(set(
                node.metadata.get('document_id')
                for node in path.nodes
                if node.metadata and 'document_id' in node.metadata
            ))
            
            # Get universal metadata for documents
            documents: List[Dict[str, Any]] = []
            
            if self.storage_manager is not None:
                for doc_id in document_ids:
                    if doc_id is not None and isinstance(doc_id, str):
                        metadata = self.storage_manager.check_document_universal_metadata(doc_id)
                        if metadata:
                            documents.append(metadata)
            
            # Add to results
            results.append({
                'path': path.to_dict(),
                'documents': documents,
                'score': path.score
            })
        
        # Sort by score (descending)
        # Use a sorting function that handles different types safely
        def get_sort_key(item: Dict[str, Any]) -> float:
            score = item.get('score')
            if score is None:
                return 0.0
            if isinstance(score, (int, float)):
                return float(score)
            return 0.0
            
        results.sort(key=get_sort_key, reverse=True)
        
        return results
    
    def process_and_execute(self, query_text: str) -> List[Dict[str, Any]]:
        """
        Process a query and execute it in one step.
        
        Args:
            query_text: The natural language query text
            
        Returns:
            List of retrieved paths and associated documents
        """
        query_plan = self.process_query(query_text)
        return self.execute_query_plan(query_plan)
