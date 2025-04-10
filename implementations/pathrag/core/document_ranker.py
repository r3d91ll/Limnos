"""
Document Ranker for PathRAG (Apollo)

This module provides a document ranking component that scores retrieved
documents based on path relevance to the query within the PathRAG framework.

The document ranker, named Apollo (god of truth and knowledge), uses various 
scoring algorithms that consider path similarity, entity overlap, relationship strength,
and semantic relevance to rank documents by their relevance to the query.
"""

import logging
import math
import re
from typing import List, Dict, Any, Tuple, Optional, Union, Set, Callable
import numpy as np
from collections import Counter, defaultdict

from .path_structures import Path as RagPath, PathNode, PathEdge
from .path_vector_store import PathVectorStore
from .path_storage_manager import PathStorageManager

# Configure logging
logger = logging.getLogger(__name__)

class DocumentRanker:
    """
    Ranks documents based on path relevance to the query.
    
    This component scores and ranks documents retrieved via path-based
    search to provide the most relevant results for a user query.
    
    Also known as Apollo in the Limnos mythology-based naming scheme.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the document ranker with configuration.
        
        Args:
            config: Configuration dictionary with options for the document ranker
        """
        self.config = config or {}
        
        # Scoring weights configuration
        self.weights = {
            'path_score': self.config.get('path_score_weight', 0.3),
            'entity_overlap': self.config.get('entity_overlap_weight', 0.2),
            'relationship_strength': self.config.get('relationship_strength_weight', 0.15),
            'semantic_similarity': self.config.get('semantic_similarity_weight', 0.25),
            'document_recency': self.config.get('document_recency_weight', 0.1)
        }
        
        # Initialize components
        self.vector_store: Optional[PathVectorStore] = None
        self.storage_manager: Optional[PathStorageManager] = None
        self.custom_scoring_functions: Dict[str, Callable] = {}
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # Path vector store for semantic similarity calculations
        self.vector_store = None
        
        # Number of paths to consider for document ranking
        self.max_paths_per_document = self.config.get('max_paths_per_document', 10)
        
        # Maximum paths per document for ranking
        # (custom_scoring_functions is initialized earlier)
        
        # Storage manager (for document metadata)
        self.storage_manager = None
    
    def set_vector_store(self, vector_store: PathVectorStore) -> None:
        """
        Set the vector store for semantic similarity calculations.
        
        Args:
            vector_store: PathVectorStore instance
        """
        self.vector_store = vector_store
    
    def set_storage_manager(self, storage_manager: PathStorageManager) -> None:
        """
        Set the storage manager for accessing document metadata.
        
        Args:
            storage_manager: PathStorageManager instance
        """
        self.storage_manager = storage_manager
    
    def register_custom_scoring_function(self, name: str, function: Callable) -> None:
        """
        Register a custom scoring function.
        
        Args:
            name: Name of the scoring function
            function: Scoring function that takes (query, paths, documents) and returns scores
        """
        self.custom_scoring_functions[name] = function
        logger.info(f"Registered custom scoring function: {name}")
    
    def _get_paths_by_document(self, paths: List[RagPath]) -> Dict[str, List[RagPath]]:
        """
        Group paths by document.
        
        Args:
            paths: List of paths
            
        Returns:
            Dictionary mapping document IDs to lists of paths
        """
        doc_paths: Dict[str, List[RagPath]] = defaultdict(list)
        
        for path in paths:
            # Extract document IDs from path nodes
            for node in path.nodes:
                if node.metadata and 'document_id' in node.metadata:
                    doc_id = node.metadata['document_id']
                    if doc_id not in doc_paths or len(doc_paths[doc_id]) < self.max_paths_per_document:
                        doc_paths[doc_id].append(path)
        
        return doc_paths
    
    def _compute_entity_overlap_score(
        self, 
        query_entities: List[Dict[str, Any]], 
        paths: List[RagPath]
    ) -> float:
        """
        Compute entity overlap score between query entities and paths.
        
        Args:
            query_entities: List of entities from the query
            paths: List of paths from a document
            
        Returns:
            Entity overlap score
        """
        if not query_entities or not paths:
            return 0.0
        
        # Extract entity texts from query
        query_entity_texts = set(entity.get('text', '').lower() for entity in query_entities)
        
        # Extract entity texts from paths
        path_entity_texts = set()
        for path in paths:
            for node in path.nodes:
                path_entity_texts.add(node.text.lower())
        
        # Calculate overlap (Jaccard similarity)
        intersection = len(query_entity_texts.intersection(path_entity_texts))
        union = len(query_entity_texts.union(path_entity_texts))
        
        return intersection / max(union, 1)
    
    def _compute_relationship_strength_score(
        self, 
        query_relationships: List[Dict[str, Any]], 
        paths: List[RagPath]
    ) -> float:
        """
        Compute relationship strength score between query relationships and paths.
        
        Args:
            query_relationships: List of relationships from the query
            paths: List of paths from a document
            
        Returns:
            Relationship strength score
        """
        if not query_relationships or not paths:
            return 0.0
        
        # Extract relationship types from query
        query_rel_types = set(rel.get('type', '') for rel in query_relationships)
        
        # Calculate relationship strength based on matches and weights
        total_strength = 0.0
        total_edges = 0
        
        for path in paths:
            for edge in path.edges:
                total_edges += 1
                if edge.type in query_rel_types:
                    # Higher weight for exact matches
                    total_strength += edge.weight * 2.0
                else:
                    # Some credit for any edge
                    total_strength += edge.weight * 0.5
        
        # Normalize by total number of edges
        return total_strength / max(total_edges, 1)
    
    def _compute_semantic_similarity_score(
        self, 
        query_text: str, 
        paths: List[RagPath]
    ) -> float:
        """
        Compute semantic similarity score between query and paths.
        
        Args:
            query_text: Query text
            paths: List of paths from a document
            
        Returns:
            Semantic similarity score
        """
        if not query_text or not paths or not self.vector_store:
            return 0.0
        
        # Get query embedding
        query_embedding = self.vector_store._get_embedding(query_text)
        
        # Calculate average similarity across paths
        similarities = []
        
        for path in paths:
            # Get path embedding
            path_embedding = self.vector_store.get_path_embedding(path)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, path_embedding)
            similarities.append(similarity)
        
        # Use average similarity
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _compute_document_recency_score(
        self, 
        document_metadata: Dict[str, Any]
    ) -> float:
        """
        Compute document recency score based on publication date.
        
        Args:
            document_metadata: Document metadata
            
        Returns:
            Document recency score
        """
        # Default recency is middle of the range if no date information
        default_recency = 0.5
        
        # Check if document has a date
        if not document_metadata:
            return default_recency
        
        # Try to get publication date (different possible fields)
        date_fields = ['date', 'publication_date', 'created_at', 'timestamp']
        date_str = None
        
        for field in date_fields:
            if field in document_metadata:
                date_str = document_metadata[field]
                break
        
        if not date_str:
            return default_recency
        
        # Simple heuristic based on year
        try:
            # Extract year (assuming format like YYYY-MM-DD or containing year)
            year_match = re.search(r'(19|20)\d{2}', str(date_str))
            if not year_match:
                return default_recency
            
            year = int(year_match.group(0))
            current_year = 2025  # Fixed current year for stable scoring
            
            # Compute recency score (higher for more recent documents)
            # Scale to [0, 1] assuming relevant documents from last 30 years
            years_ago = current_year - year
            if years_ago < 0:
                # Future date (shouldn't happen, but just in case)
                return 1.0
            elif years_ago > 30:
                # Very old document
                return 0.1
            else:
                # Linear decay over 30 years
                return 1.0 - (years_ago / 30.0 * 0.9)
            
        except (ValueError, TypeError):
            return default_recency
    
    def _aggregate_path_scores(self, paths: List[RagPath]) -> float:
        """
        Aggregate scores from multiple paths.
        
        Args:
            paths: List of paths
            
        Returns:
            Aggregated path score
        """
        if not paths:
            return 0.0
        
        # Sort paths by score (descending)
        sorted_paths = sorted(paths, key=lambda p: p.score, reverse=True)
        
        # Use diminishing returns for additional paths
        total_score = 0.0
        for i, path in enumerate(sorted_paths):
            # Apply diminishing returns factor
            factor = math.exp(-0.3 * i)  # Exponential decay
            total_score += path.score * factor
        
        # Normalize to [0, 1]
        max_possible = sum(math.exp(-0.3 * i) for i in range(len(paths)))
        return total_score / max_possible if max_possible > 0 else 0.0
    
    def _apply_custom_scoring_functions(
        self,
        query: Dict[str, Any],
        documents: Dict[str, Dict[str, Any]],
        doc_paths: Dict[str, List[RagPath]],
        scores: Dict[str, float]
    ) -> None:
        """
        Apply registered custom scoring functions and update scores.
        
        Args:
            query: Query dictionary
            documents: Dictionary mapping document IDs to document metadata
            doc_paths: Dictionary mapping document IDs to lists of paths
            scores: Dictionary mapping document IDs to scores (modified in-place)
        """
        for name, func in self.custom_scoring_functions.items():
            try:
                # Apply custom scoring function
                custom_scores = func(query, doc_paths, documents)
                
                # Update scores (custom scores assumed to be normalized to [0, 1])
                weight = self.config.get(f'{name}_weight', 0.1)
                for doc_id, custom_score in custom_scores.items():
                    if doc_id in scores:
                        # Adjust score by custom function weight
                        # We assume other weights are already normalized to sum to 1.0
                        # So we apply this as a separate factor
                        scores[doc_id] = scores[doc_id] * (1.0 - weight) + custom_score * weight
                
            except Exception as e:
                logger.error(f"Error in custom scoring function '{name}': {e}")
    
    def rank_documents(
        self,
        query: Dict[str, Any],
        paths: List[RagPath],
        top_k: int = 10
    ) -> List[Tuple[str, float, List[RagPath]]]:
        """
        Rank documents based on path relevance to the query.
        
        Args:
            query: Query dictionary with 'original_query', 'processed_query', 'entities', etc.
            paths: List of paths retrieved for the query
            top_k: Number of top documents to return
            
        Returns:
            List of (document_id, score, relevant_paths) tuples sorted by score
        """
        # Group paths by document
        doc_paths = self._get_paths_by_document(paths)
        
        # Get document metadata
        documents: Dict[str, Dict[str, Any]] = {}
        if self.storage_manager:
            for doc_id in doc_paths.keys():
                metadata = self.storage_manager.check_document_universal_metadata(doc_id)
                if metadata:
                    documents[doc_id] = metadata
        
        # Compute scores for each document
        scores: Dict[str, float] = {}
        
        query_entities = query.get('entities', [])
        query_relationships = query.get('relationships', [])
        query_text = query.get('processed_query', query.get('original_query', ''))
        
        # Fix potential incompatible type issue by ensuring we have the right type
        if not isinstance(doc_paths, dict):
            # Convert to dictionary format if it's not already
            doc_paths_dict: Dict[str, List[RagPath]] = {}
            for path in doc_paths:
                for node in path.nodes:
                    if node.metadata and 'document_id' in node.metadata:
                        doc_id = node.metadata['document_id']
                        if doc_id not in doc_paths_dict:
                            doc_paths_dict[doc_id] = []
                        doc_paths_dict[doc_id].append(path)
            doc_paths = doc_paths_dict
            
        for doc_id, doc_path_list in doc_paths.items():
            # Get document paths as a list for scoring methods
            paths_list: List[RagPath] = doc_path_list
            
            # Safety check - ensure we're working with a list
            if not isinstance(paths_list, list):
                logger.warning(f"Expected a list of paths, got {type(paths_list)}")
                # Create an empty list as a fallback
                paths_list = []
            
            # Compute individual score components
            path_score = self._aggregate_path_scores(paths_list)
            entity_overlap = self._compute_entity_overlap_score(query_entities, paths_list)
            relationship_strength = self._compute_relationship_strength_score(query_relationships, paths_list)
            semantic_similarity = self._compute_semantic_similarity_score(query_text, paths_list)
            
            # Get document recency score
            doc_metadata: Dict[str, Any] = documents.get(doc_id, {})
            recency_score = self._compute_document_recency_score(doc_metadata)
            
            # Compute weighted score
            # Ensure all score components are properly initialized before using them
            path_score_val = 0.0 if path_score is None else path_score
            entity_overlap_val = 0.0 if entity_overlap is None else entity_overlap
            relationship_strength_val = 0.0 if relationship_strength is None else relationship_strength
            semantic_similarity_val = 0.0 if semantic_similarity is None else semantic_similarity
            recency_score_val = 0.0 if recency_score is None else recency_score
            
            score = (
                path_score_val * self.weights['path_score'] +
                entity_overlap_val * self.weights['entity_overlap'] +
                relationship_strength_val * self.weights['relationship_strength'] +
                semantic_similarity_val * self.weights['semantic_similarity'] +
                recency_score_val * self.weights['document_recency']
            )
            
            scores[doc_id] = score
        
        # Apply custom scoring functions if any
        self._apply_custom_scoring_functions(query, documents, doc_paths, scores)
        
        # Sort documents by score (descending)
        ranked_docs = [
            (doc_id, score, doc_paths[doc_id])
            for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Return top-k
        return ranked_docs[:top_k]
    
    def format_ranked_documents(
        self,
        ranked_docs: List[Tuple[str, float, List[RagPath]]],
        include_paths: bool = True,
        include_score_details: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Format ranked documents for presentation.
        
        Args:
            ranked_docs: List of (document_id, score, relevant_paths) tuples
            include_paths: Whether to include path details
            include_score_details: Whether to include score component details
            
        Returns:
            List of formatted document dictionaries
        """
        formatted_docs = []
        
        for doc_id, score, paths in ranked_docs:
            # Get document metadata
            doc_metadata = {}
            if self.storage_manager:
                metadata = self.storage_manager.check_document_universal_metadata(doc_id)
                if metadata:
                    doc_metadata = metadata
            
            # Prepare document result
            doc_result = {
                'document_id': doc_id,
                'score': score,
                'metadata': doc_metadata
            }
            
            # Add paths if requested
            if include_paths:
                doc_result['paths'] = [path.to_dict() for path in paths]
            
            # Add score details if requested
            if include_score_details and paths:
                query_entities = [] if not hasattr(self, '_last_query') else self._last_query.get('entities', [])
                query_relationships = [] if not hasattr(self, '_last_query') else self._last_query.get('relationships', [])
                query_text = '' if not hasattr(self, '_last_query') else self._last_query.get('processed_query', '')
                
                doc_result['score_details'] = {
                    'path_score': self._aggregate_path_scores(paths),
                    'entity_overlap': self._compute_entity_overlap_score(query_entities, paths),
                    'relationship_strength': self._compute_relationship_strength_score(query_relationships, paths),
                    'semantic_similarity': self._compute_semantic_similarity_score(query_text, paths),
                    'document_recency': self._compute_document_recency_score(doc_metadata),
                    'weights': self.weights
                }
            
            formatted_docs.append(doc_result)
        
        return formatted_docs
    
    def rank_and_format_documents(
        self,
        query: Dict[str, Any],
        paths: List[RagPath],
        top_k: int = 10,
        include_paths: bool = True,
        include_score_details: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Rank and format documents in one step.
        
        Args:
            query: Query dictionary
            paths: List of paths
            top_k: Number of top documents to return
            include_paths: Whether to include path details
            include_score_details: Whether to include score component details
            
        Returns:
            List of formatted document dictionaries
        """
        # Save query for score details
        self._last_query = query
        
        # Rank documents
        ranked_docs = self.rank_documents(query, paths, top_k)
        
        # Format results
        return self.format_ranked_documents(ranked_docs, include_paths, include_score_details)
