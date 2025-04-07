"""
Relevance Search

Implements relevance-based search algorithms for knowledge graphs.
"""

import logging
import heapq
from typing import Dict, List, Set, Any, Optional, Callable, Tuple, Union, DefaultDict
import networkx as nx
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class RelevanceSearch:
    """
    Provides relevance-based search capabilities for knowledge graphs.
    
    This class implements algorithms for finding relevant nodes, subgraphs,
    and paths based on semantic relevance to a query or other relevance metrics.
    """
    
    def __init__(self, 
                 max_results: int = 100, 
                 relevance_threshold: float = 0.5,
                 use_edge_weights: bool = True):
        """
        Initialize the relevance search engine.
        
        Args:
            max_results: Maximum number of results to return
            relevance_threshold: Minimum relevance score (0-1) for results
            use_edge_weights: Whether to consider edge weights in relevance calculations
        """
        self.max_results = max_results
        self.relevance_threshold = relevance_threshold
        self.use_edge_weights = use_edge_weights
        logger.info(f"Initialized RelevanceSearch with max_results={max_results}, " 
                   f"relevance_threshold={relevance_threshold}")
    
    def vector_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            float: Cosine similarity (-1 to 1)
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)  # Avoid division by zero
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        return float(similarity)
    
    def search_by_embedding(self,
                           graph: nx.Graph,
                           query_vector: np.ndarray,
                           embedding_attr: str = 'embedding',
                           max_results: Optional[int] = None,
                           min_similarity: Optional[float] = None) -> List[Tuple[Any, float]]:
        """
        Search for nodes based on embedding similarity to a query vector.
        
        Args:
            graph: NetworkX graph to search
            query_vector: Query embedding vector
            embedding_attr: Node attribute containing embedding vectors
            max_results: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List[Tuple[Any, float]]: List of (node, similarity) tuples
        """
        max_results = max_results if max_results is not None else self.max_results
        min_similarity = min_similarity if min_similarity is not None else self.relevance_threshold
        
        results = []
        
        # Search through all nodes
        for node in graph.nodes():
            # Get node embedding
            node_data = graph.nodes[node]
            if embedding_attr not in node_data:
                continue
            
            embedding = node_data[embedding_attr]
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Calculate similarity
            similarity = self.vector_similarity(query_vector, embedding)
            
            # Add to results if above threshold
            if similarity >= min_similarity:
                results.append((node, similarity))
        
        # Sort by similarity (descending) and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        if max_results:
            results = results[:max_results]
        
        logger.debug(f"Found {len(results)} nodes with embedding similarity")
        return results
    
    def search_by_attribute(self,
                          graph: nx.Graph,
                          attr_name: str,
                          attr_value: Any,
                          exact_match: bool = True,
                          max_results: Optional[int] = None) -> List[Any]:
        """
        Search for nodes with a specific attribute value.
        
        Args:
            graph: NetworkX graph to search
            attr_name: Attribute name to search
            attr_value: Attribute value to match
            exact_match: Whether to require exact matches
            max_results: Maximum number of results to return
            
        Returns:
            List[Any]: List of matching nodes
        """
        max_results = max_results if max_results is not None else self.max_results
        
        results = []
        
        # Search through all nodes
        for node in graph.nodes():
            # Get node attributes
            node_data = graph.nodes[node]
            
            # Check for attribute
            if attr_name in node_data:
                node_value = node_data[attr_name]
                
                # Match based on exact_match flag
                if exact_match and node_value == attr_value:
                    results.append(node)
                elif not exact_match and attr_value in node_value:
                    results.append(node)
        
        # Limit results
        if max_results and len(results) > max_results:
            results = results[:max_results]
        
        logger.debug(f"Found {len(results)} nodes with matching attribute {attr_name}")
        return results
    
    def personalized_pagerank(self,
                             graph: nx.Graph,
                             seed_nodes: List[Any],
                             alpha: float = 0.85,
                             max_iterations: int = 100,
                             tolerance: float = 1.0e-6,
                             max_results: Optional[int] = None) -> Dict[Any, float]:
        """
        Compute personalized PageRank scores starting from seed nodes.
        
        Args:
            graph: NetworkX graph to analyze
            seed_nodes: List of starting nodes for personalization
            alpha: Damping parameter (default: 0.85)
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            max_results: Maximum number of results to return
            
        Returns:
            Dict[Any, float]: Map of node -> PageRank score
        """
        max_results = max_results if max_results is not None else self.max_results
        
        # Create personalization dict with equal weights for seed nodes
        personalization = {}
        for node in seed_nodes:
            if node in graph:
                personalization[node] = 1.0 / len(seed_nodes)
        
        # If no valid seed nodes, return empty result
        if not personalization:
            logger.warning("No valid seed nodes for personalized PageRank")
            return {}
        
        # Compute PageRank
        try:
            pagerank = nx.pagerank(
                graph,
                alpha=alpha,
                personalization=personalization,
                max_iter=max_iterations,
                tol=tolerance,
                weight='weight' if self.use_edge_weights else None
            )
            
            # Sort by score (descending) and limit results
            sorted_results = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            if max_results:
                sorted_results = sorted_results[:max_results]
            
            # Convert back to dict
            results = dict(sorted_results)
            
            logger.debug(f"Computed personalized PageRank scores for {len(results)} nodes")
            return results
        except Exception as e:
            logger.error(f"Error computing personalized PageRank: {e}")
            return {}
    
    def spread_activation_search(self,
                               graph: nx.Graph,
                               seed_nodes: Dict[Any, float],
                               decay_factor: float = 0.5,
                               activation_threshold: float = 0.01,
                               max_iterations: int = 10,
                               max_results: Optional[int] = None) -> Dict[Any, float]:
        """
        Perform spreading activation search from seed nodes.
        
        This algorithm propagates activation values through the graph,
        with activation decaying as it spreads away from seed nodes.
        
        Args:
            graph: NetworkX graph to search
            seed_nodes: Dict mapping seed nodes to initial activation values
            decay_factor: Factor by which activation decays (0-1)
            activation_threshold: Minimum activation to continue spreading
            max_iterations: Maximum number of spreading iterations
            max_results: Maximum number of results to return
            
        Returns:
            Dict[Any, float]: Map of node -> activation value
        """
        max_results = max_results if max_results is not None else self.max_results
        
        # Filter seed nodes to those in graph
        seed_nodes = {node: value for node, value in seed_nodes.items() if node in graph}
        
        # If no valid seed nodes, return empty result
        if not seed_nodes:
            logger.warning("No valid seed nodes for spreading activation")
            return {}
        
        # Initialize activations
        activations = dict(seed_nodes)  # Copy initial activations
        
        # Track nodes to process in each iteration
        active_nodes = set(seed_nodes.keys())
        
        # Spreading activation iterations
        for iteration in range(max_iterations):
            # No more active nodes to process
            if not active_nodes:
                break
            
            # Prepare next iteration's active nodes
            next_active_nodes: Set[Any] = set()
            new_activations: DefaultDict[Any, float] = defaultdict(float)
            
            # Process each active node
            for node in active_nodes:
                # Get current activation
                current_activation = activations.get(node, 0.0)
                
                # Skip if below threshold
                if current_activation < activation_threshold:
                    continue
                
                # Calculate outgoing activation
                outgoing_activation = current_activation * decay_factor
                
                # Count neighbors for normalization
                neighbors = list(graph.neighbors(node))
                if not neighbors:
                    continue
                
                # Distribute activation to neighbors
                activation_per_neighbor = outgoing_activation / len(neighbors)
                
                for neighbor in neighbors:
                    # Apply edge weight if enabled
                    edge_weight = 1.0
                    if self.use_edge_weights:
                        edge_data = graph.get_edge_data(node, neighbor)
                        if 'weight' in edge_data:
                            edge_weight = edge_data['weight']
                    
                    # Calculate neighbor activation
                    neighbor_activation = activation_per_neighbor * edge_weight
                    
                    # Only propagate if above threshold
                    if neighbor_activation >= activation_threshold:
                        new_activations[neighbor] += neighbor_activation
                        next_active_nodes.add(neighbor)
            
            # Update activations
            for node, activation in new_activations.items():
                if node in activations:
                    activations[node] += activation
                else:
                    activations[node] = activation
            
            # Update active nodes for next iteration
            active_nodes = next_active_nodes
            
            logger.debug(f"Spreading activation iteration {iteration+1}: "
                        f"{len(active_nodes)} active nodes")
        
        # Sort by activation (descending) and limit results
        sorted_results = sorted(activations.items(), key=lambda x: x[1], reverse=True)
        if max_results:
            sorted_results = sorted_results[:max_results]
        
        # Convert back to dict
        results = dict(sorted_results)
        
        logger.debug(f"Spreading activation found {len(results)} activated nodes")
        return results
    
    def search_by_weighted_attributes(self,
                                    graph: nx.Graph,
                                    attr_weights: Dict[str, Tuple[Any, float]],
                                    max_results: Optional[int] = None,
                                    min_score: float = 0.0) -> List[Tuple[Any, float]]:
        """
        Search for nodes based on multiple weighted attributes.
        
        Args:
            graph: NetworkX graph to search
            attr_weights: Dict mapping attribute names to (value, weight) tuples
            max_results: Maximum number of results to return
            min_score: Minimum combined score to include in results
            
        Returns:
            List[Tuple[Any, float]]: List of (node, score) tuples
        """
        max_results = max_results if max_results is not None else self.max_results
        
        results = []
        
        # Normalize weights
        total_weight = sum(weight for _, weight in attr_weights.values())
        normalized_weights = {
            attr: (value, weight / total_weight) 
            for attr, (value, weight) in attr_weights.items()
        }
        
        # Score each node
        for node in graph.nodes():
            node_data = graph.nodes[node]
            score = 0.0
            
            # Calculate score based on attribute matches
            for attr, (value, weight) in normalized_weights.items():
                if attr in node_data:
                    node_value = node_data[attr]
                    
                    # Binary match
                    if node_value == value:
                        score += weight
            
            # Add to results if above threshold
            if score >= min_score:
                results.append((node, score))
        
        # Sort by score (descending) and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        if max_results:
            results = results[:max_results]
        
        logger.debug(f"Found {len(results)} nodes with weighted attribute matches")
        return results
    
    def semantic_expansion(self,
                         graph: nx.Graph,
                         seed_nodes: List[Any],
                         expansion_factor: float = 0.7,
                         max_depth: int = 2,
                         max_nodes: Optional[int] = None,
                         embedding_attr: str = 'embedding',
                         query_vector: Optional[np.ndarray] = None) -> List[Any]:
        """
        Expand a set of seed nodes based on semantic relevance.
        
        This method expands from seed nodes to semantically similar neighbors,
        with expansion guided by embedding similarity if query_vector is provided.
        
        Args:
            graph: NetworkX graph to search
            seed_nodes: Starting nodes for expansion
            expansion_factor: Factor controlling how many neighbors to explore (0-1)
            max_depth: Maximum expansion depth
            max_nodes: Maximum nodes in expanded set
            embedding_attr: Node attribute containing embeddings
            query_vector: Optional query vector for relevance guidance
            
        Returns:
            List[Any]: Expanded list of nodes
        """
        max_nodes = max_nodes if max_nodes is not None else self.max_results
        
        # Filter seed nodes to those in graph
        seed_nodes = [node for node in seed_nodes if node in graph]
        
        # Initialize expansion set with seed nodes
        expanded_nodes = set(seed_nodes)
        frontier = set(seed_nodes)
        
        # Helper function for semantic guidance
        def node_relevance(node: Any) -> float:
            # If no query vector, all nodes equally relevant
            if query_vector is None:
                return 1.0
            
            # Get node embedding
            node_data = graph.nodes[node]
            if embedding_attr not in node_data:
                return 0.0
            
            embedding = node_data[embedding_attr]
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Calculate similarity
            return self.vector_similarity(query_vector, embedding)
        
        # Track node relevance scores
        relevance_scores = {node: node_relevance(node) for node in seed_nodes}
        
        # Perform semantic expansion
        for depth in range(max_depth):
            if not frontier or len(expanded_nodes) >= max_nodes:
                break
            
            next_frontier = set()
            
            # Process each frontier node
            for node in frontier:
                current_node_relevance = relevance_scores.get(node, 0.0)
                
                # Get neighbors
                neighbors = list(graph.neighbors(node))
                
                # Skip if no neighbors
                if not neighbors:
                    continue
                
                # Calculate expansion limit based on relevance
                expansion_limit = max(1, int(len(neighbors) * expansion_factor * current_node_relevance))
                
                # Score and sort neighbors by relevance
                neighbor_scores = []
                for neighbor in neighbors:
                    if neighbor in expanded_nodes:
                        continue
                    
                    score = node_relevance(neighbor)
                    relevance_scores[neighbor] = score
                    neighbor_scores.append((neighbor, score))
                
                # Sort by relevance
                neighbor_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Add top neighbors to next frontier
                for neighbor, _ in neighbor_scores[:expansion_limit]:
                    next_frontier.add(neighbor)
                    expanded_nodes.add(neighbor)
                    
                    # Check if we've reached the max nodes
                    if len(expanded_nodes) >= max_nodes:
                        break
            
            # Update frontier for next iteration
            frontier = next_frontier
            
            logger.debug(f"Semantic expansion depth {depth+1}: "
                        f"{len(frontier)} frontier nodes, {len(expanded_nodes)} total nodes")
        
        # Sort by relevance score
        result_nodes = list(expanded_nodes)
        result_nodes.sort(key=lambda n: relevance_scores.get(n, 0.0), reverse=True)
        
        logger.debug(f"Semantic expansion found {len(result_nodes)} relevant nodes")
        return result_nodes
