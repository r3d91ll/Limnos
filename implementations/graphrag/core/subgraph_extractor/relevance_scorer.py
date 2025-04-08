"""
Relevance Scorer for Nodes and Edges in Knowledge Graphs.

This module provides functionality for scoring the relevance of nodes and edges
in a knowledge graph with respect to a query or context.
"""

import logging
import math
from typing import Dict, List, Set, Any, Optional, Callable, Tuple, Union
import networkx as nx
import numpy as np
from scipy import spatial  # type: ignore # Missing stubs for scipy

logger = logging.getLogger(__name__)

class NodeEdgeScorer:
    """
    Scores the relevance of nodes and edges in a knowledge graph based on various criteria.
    """
    
    def __init__(self, 
                embedding_attr: str = 'embedding',
                text_attr: str = 'text',
                importance_attr: str = 'importance',
                alpha: float = 0.5,
                beta: float = 0.3,
                gamma: float = 0.2,
                min_score: float = 0.0,
                normalize_scores: bool = True):
        """
        Initialize the relevance scorer with configurable parameters.
        
        Args:
            embedding_attr: Attribute name for node/edge embeddings
            text_attr: Attribute name for node/edge text content
            importance_attr: Attribute name for pre-calculated importance
            alpha: Weight for semantic similarity in scoring
            beta: Weight for structural importance in scoring
            gamma: Weight for pre-defined importance in scoring
            min_score: Minimum score threshold for relevance
            normalize_scores: Whether to normalize scores to [0,1] range
        """
        self.embedding_attr = embedding_attr
        self.text_attr = text_attr
        self.importance_attr = importance_attr
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_score = min_score
        self.normalize_scores = normalize_scores
        
        logger.info(f"Initialized NodeEdgeScorer with alpha={alpha}, beta={beta}, gamma={gamma}")
    
    def calculate_semantic_similarity(self, 
                                     query_embedding: np.ndarray, 
                                     entity_embedding: np.ndarray) -> float:
        """
        Calculate semantic similarity between query and entity embeddings.
        
        Args:
            query_embedding: Query embedding vector
            entity_embedding: Entity embedding vector
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Ensure both embeddings are normalized
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        entity_norm = entity_embedding / np.linalg.norm(entity_embedding)
        
        # Calculate cosine similarity
        similarity = 1 - spatial.distance.cosine(query_norm, entity_norm)
        
        # Ensure score is between 0 and 1 and explicitly return float
        result: float = max(0.0, min(1.0, float(similarity)))
        return result
    
    def calculate_structural_importance(self, 
                                       graph: nx.Graph, 
                                       node: Any, 
                                       method: str = 'degree') -> float:
        """
        Calculate structural importance of a node based on graph topology.
        
        Args:
            graph: NetworkX graph containing the node
            node: The node to calculate importance for
            method: Method for calculating importance (degree, centrality, pagerank)
            
        Returns:
            float: Importance score between 0 and 1
        """
        if method == 'degree':
            # Use node degree (normalized by max degree in graph)
            # Safe way to get degree information that works with all NetworkX versions
            degree_values: List[int] = []
            
            # Get degree for each node individually to avoid any iteration issues
            for n in graph.nodes():
                try:
                    # Get degree for this specific node
                    deg = graph.degree(n)
                    # Handle case where degree might be a view or an int
                    if isinstance(deg, int):
                        degree_values.append(deg)
                    else:
                        # If it's not an int, try to get the integer value
                        # This handles the case of node attributes in degree views
                        degree_values.append(1)  # Default fallback
                except (TypeError, ValueError, AttributeError):
                    # If we can't get a degree, use 1 as a safe default
                    degree_values.append(1)
                    
            # Calculate max degree safely
            max_degree = max(degree_values) if degree_values else 1
            
            # Get degree for the specific node safely
            try:
                # Try to get degree directly for this node
                node_deg = graph.degree(node)
                # Ensure we have an int
                node_degree = node_deg if isinstance(node_deg, int) else 0
            except (TypeError, ValueError, AttributeError):
                # Default to 0 if we can't get the degree
                node_degree = 0
            return float(node_degree) / float(max_degree)
        
        elif method == 'centrality':
            # Use betweenness centrality
            if not hasattr(self, '_betweenness_cache'):
                # Calculate once and cache for efficiency
                self._betweenness_cache = nx.betweenness_centrality(graph)
            
            betweenness_result: float = self._betweenness_cache.get(node, 0.0)
            return betweenness_result
        
        elif method == 'pagerank':
            # Use PageRank
            if not hasattr(self, '_pagerank_cache'):
                # Calculate once and cache for efficiency
                self._pagerank_cache = nx.pagerank(graph)
            
            pagerank_result: float = self._pagerank_cache.get(node, 0.0)
            return pagerank_result
        
        else:
            logger.warning(f"Unknown structural importance method: {method}, using degree")
            return self.calculate_structural_importance(graph, node, 'degree')
    
    def get_predefined_importance(self, entity: Any, attr_dict: Dict) -> float:
        """
        Get pre-defined importance from entity attributes.
        
        Args:
            entity: The entity (node or edge)
            attr_dict: Dictionary of entity attributes
            
        Returns:
            float: Importance score between 0 and 1
        """
        importance = attr_dict.get(self.importance_attr, 0.0)
        
        # If it's not a number, try to convert or default to 0
        if not isinstance(importance, (int, float)):
            try:
                importance = float(importance)
            except (ValueError, TypeError):
                importance = 0.0
        
        # Ensure it's between 0 and 1
        importance_result: float = max(0.0, min(1.0, float(importance)))
        return importance_result
    
    def score_node(self, 
                  graph: nx.Graph, 
                  node: Any, 
                  query_embedding: Optional[np.ndarray] = None,
                  structural_method: str = 'degree') -> float:
        """
        Calculate the overall relevance score for a node.
        
        Args:
            graph: NetworkX graph containing the node
            node: The node to score
            query_embedding: Query embedding for semantic similarity
            structural_method: Method for structural importance
            
        Returns:
            float: Overall relevance score between 0 and 1
        """
        node_attrs = graph.nodes[node]
        scores = []
        weights = []
        
        # Semantic similarity component
        if query_embedding is not None and self.embedding_attr in node_attrs:
            node_embedding = node_attrs[self.embedding_attr]
            if isinstance(node_embedding, list) or isinstance(node_embedding, np.ndarray):
                semantic_score = self.calculate_semantic_similarity(
                    query_embedding, np.array(node_embedding))
                scores.append(semantic_score)
                weights.append(self.alpha)
        
        # Structural importance component
        structural_score = self.calculate_structural_importance(
            graph, node, structural_method)
        scores.append(structural_score)
        weights.append(self.beta)
        
        # Predefined importance component
        predefined_score = self.get_predefined_importance(node, node_attrs)
        scores.append(predefined_score)
        weights.append(self.gamma)
        
        # Weighted average of available scores
        if sum(weights) == 0:
            return 0.0
            
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        # Apply minimum threshold
        if weighted_score < self.min_score:
            return 0.0
            
        return weighted_score
    
    def score_edge(self, 
                  graph: nx.Graph, 
                  edge: Tuple[Any, Any], 
                  query_embedding: Optional[np.ndarray] = None) -> float:
        """
        Calculate the overall relevance score for an edge.
        
        Args:
            graph: NetworkX graph containing the edge
            edge: The edge to score (u, v)
            query_embedding: Query embedding for semantic similarity
            
        Returns:
            float: Overall relevance score between 0 and 1
        """
        u, v = edge
        edge_attrs = graph.edges[edge]
        scores = []
        weights = []
        
        # Semantic similarity component (if edge has embedding)
        if query_embedding is not None and self.embedding_attr in edge_attrs:
            edge_embedding = edge_attrs[self.embedding_attr]
            if isinstance(edge_embedding, list) or isinstance(edge_embedding, np.ndarray):
                semantic_score = self.calculate_semantic_similarity(
                    query_embedding, np.array(edge_embedding))
                scores.append(semantic_score)
                weights.append(self.alpha)
        
        # Edge importance based on endpoint nodes
        node_u_score = self.score_node(graph, u, query_embedding)
        node_v_score = self.score_node(graph, v, query_embedding)
        endpoint_score = (node_u_score + node_v_score) / 2
        scores.append(endpoint_score)
        weights.append(self.beta)
        
        # Predefined importance component
        predefined_score = self.get_predefined_importance(edge, edge_attrs)
        scores.append(predefined_score)
        weights.append(self.gamma)
        
        # Weighted average of available scores
        if sum(weights) == 0:
            return 0.0
            
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        # Apply minimum threshold
        if weighted_score < self.min_score:
            return 0.0
            
        return weighted_score
    
    def score_graph(self, 
                   graph: nx.Graph, 
                   query_embedding: Optional[np.ndarray] = None,
                   structural_method: str = 'degree') -> Tuple[Dict[Any, float], Dict[Tuple[Any, Any], float]]:
        """
        Score all nodes and edges in a graph based on relevance to query.
        
        Args:
            graph: NetworkX graph to score
            query_embedding: Query embedding for semantic similarity
            structural_method: Method for structural importance
            
        Returns:
            Tuple[Dict[Any, float], Dict[Tuple[Any, Any], float]]: 
                (node_scores, edge_scores) dictionaries
        """
        # Score all nodes
        node_scores = {}
        for node in graph.nodes():
            node_scores[node] = self.score_node(
                graph, node, query_embedding, structural_method)
        
        # Score all edges
        edge_scores = {}
        for u, v in graph.edges():
            edge_scores[(u, v)] = self.score_edge(
                graph, (u, v), query_embedding)
        
        # Normalize scores if requested
        if self.normalize_scores:
            node_scores = self._normalize_scores(node_scores)
            edge_scores = self._normalize_scores(edge_scores)
        
        return node_scores, edge_scores
    
    def _normalize_scores(self, scores: Dict[Any, float]) -> Dict[Any, float]:
        """
        Normalize a dictionary of scores to [0,1] range.
        
        Args:
            scores: Dictionary of entity -> score
            
        Returns:
            Dict[Any, float]: Normalized scores
        """
        if not scores:
            return {}
            
        min_score = min(scores.values())
        max_score = max(scores.values())
        
        # If all scores are the same, return original
        if max_score == min_score:
            return scores
            
        # Normalize to [0,1]
        normalized = {}
        for entity, score in scores.items():
            normalized[entity] = (score - min_score) / (max_score - min_score)
            
        return normalized
