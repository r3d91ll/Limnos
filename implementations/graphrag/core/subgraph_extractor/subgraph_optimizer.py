"""
Subgraph Optimization Techniques.

This module provides algorithms for optimizing subgraphs to enhance quality,
relevance, and structural properties while maintaining the most important information.
"""

import logging
import math
import random
from typing import Dict, List, Set, Any, Optional, Callable, Tuple, Union
import networkx as nx
import numpy as np

from .relevance_scorer import NodeEdgeScorer

logger = logging.getLogger(__name__)

class SubgraphOptimizer:
    """
    Provides techniques for optimizing extracted subgraphs.
    """
    
    def __init__(self, 
                 relevance_weight: float = 0.6,
                 diversity_weight: float = 0.2,
                 connectivity_weight: float = 0.2,
                 max_iterations: int = 10,
                 improvement_threshold: float = 0.01,
                 random_seed: Optional[int] = None):
        """
        Initialize the subgraph optimizer with configurable parameters.
        
        Args:
            relevance_weight: Weight for relevance score in optimization
            diversity_weight: Weight for diversity score in optimization
            connectivity_weight: Weight for connectivity score in optimization
            max_iterations: Maximum iterations for optimization algorithms
            improvement_threshold: Minimum improvement to continue optimization
            random_seed: Random seed for reproducibility
        """
        self.relevance_weight = relevance_weight
        self.diversity_weight = diversity_weight
        self.connectivity_weight = connectivity_weight
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        logger.info(f"Initialized SubgraphOptimizer with relevance_weight={relevance_weight}, "
                   f"diversity_weight={diversity_weight}, connectivity_weight={connectivity_weight}")
    
    def optimize_relevance(self, 
                          graph: nx.Graph, 
                          node_scores: Dict[Any, float],
                          edge_scores: Dict[Tuple[Any, Any], float],
                          seed_nodes: List[Any],
                          max_swaps: int = 20) -> nx.Graph:
        """
        Optimize subgraph for maximum relevance by swapping nodes.
        
        Args:
            graph: NetworkX graph to optimize
            node_scores: Dictionary of node -> relevance score
            edge_scores: Dictionary of edge -> relevance score
            seed_nodes: Important nodes to preserve
            max_swaps: Maximum number of node swaps to perform
            
        Returns:
            nx.Graph: Optimized subgraph
        """
        # Create a copy of the graph
        optimized_graph = graph.copy()
        
        # Initialize candidate nodes (nodes outside the current graph)
        original_graph = self._get_original_graph_from_scores(node_scores, edge_scores)
        candidate_nodes = [node for node in original_graph.nodes 
                         if node not in optimized_graph.nodes]
        
        # Sort candidates by score
        candidate_nodes.sort(key=lambda n: node_scores.get(n, 0), reverse=True)
        
        # Calculate initial total relevance
        initial_relevance = self._calculate_total_relevance(
            optimized_graph, node_scores, edge_scores)
        
        # Track nodes that can be swapped (not seed nodes)
        swappable_nodes = [node for node in optimized_graph.nodes 
                          if node not in seed_nodes]
        
        # Sort by score (ascending)
        swappable_nodes.sort(key=lambda n: node_scores.get(n, 0))
        
        # Perform swaps
        num_swaps = 0
        for i in range(min(len(swappable_nodes), len(candidate_nodes), max_swaps)):
            if not swappable_nodes or not candidate_nodes:
                break
                
            # Get lowest-scoring node in graph and highest-scoring candidate
            node_to_remove = swappable_nodes[0]
            node_to_add = candidate_nodes[0]
            
            # Calculate relevance change if we swap these nodes
            temp_graph = optimized_graph.copy()
            
            # Remove node
            node_neighbors = list(temp_graph.neighbors(node_to_remove))
            node_attrs = temp_graph.nodes[node_to_remove].copy()
            temp_graph.remove_node(node_to_remove)
            
            # Add new node with connections to neighbors of removed node
            temp_graph.add_node(node_to_add, **original_graph.nodes.get(node_to_add, {}))
            
            # Connect to same neighbors as removed node
            for neighbor in node_neighbors:
                if neighbor in temp_graph:
                    # Use original edge attributes if available
                    if original_graph.has_edge(node_to_add, neighbor):
                        temp_graph.add_edge(
                            node_to_add, neighbor, **original_graph.edges[node_to_add, neighbor])
                    else:
                        temp_graph.add_edge(node_to_add, neighbor, weight=1.0)
            
            # Calculate new relevance
            new_relevance = self._calculate_total_relevance(
                temp_graph, node_scores, edge_scores)
            
            # If improvement, make the swap
            if new_relevance > initial_relevance:
                optimized_graph = temp_graph
                initial_relevance = new_relevance
                num_swaps += 1
                
                # Update lists
                swappable_nodes.pop(0)
                candidate_nodes.pop(0)
                
                # Add new node to swappable
                if node_to_add not in seed_nodes:
                    swappable_nodes.append(node_to_add)
                    swappable_nodes.sort(key=lambda n: node_scores.get(n, 0))
            else:
                # Remove these nodes from consideration
                swappable_nodes.pop(0)
                candidate_nodes.pop(0)
        
        logger.info(f"Relevance optimization performed {num_swaps} node swaps")
        
        return optimized_graph
    
    def _calculate_total_relevance(self, 
                                 graph: nx.Graph, 
                                 node_scores: Dict[Any, float],
                                 edge_scores: Dict[Tuple[Any, Any], float]) -> float:
        """
        Calculate total relevance score for a graph.
        
        Args:
            graph: NetworkX graph to calculate score for
            node_scores: Dictionary of node -> relevance score
            edge_scores: Dictionary of edge -> relevance score
            
        Returns:
            float: Total relevance score
        """
        # Sum node scores
        node_score_sum = sum(node_scores.get(node, 0.0) for node in graph.nodes)
        
        # Sum edge scores
        edge_score_sum = sum(edge_scores.get(edge, 0.0) 
                            for edge in graph.edges)
        
        # Weighted combination
        node_weight = 0.7  # Weight for nodes vs edges
        total_score = (node_weight * node_score_sum + 
                     (1 - node_weight) * edge_score_sum)
        
        return total_score
    
    def _get_original_graph_from_scores(self, 
                                      node_scores: Dict[Any, float],
                                      edge_scores: Dict[Tuple[Any, Any], float]) -> nx.Graph:
        """
        Reconstruct an approximation of the original graph from score dictionaries.
        
        Args:
            node_scores: Dictionary of node -> relevance score
            edge_scores: Dictionary of edge -> relevance score
            
        Returns:
            nx.Graph: Reconstructed graph
        """
        # Create a new graph with proper type annotation
        original_graph: nx.Graph = nx.Graph()
        
        # Add nodes with dummy attributes
        for node in node_scores.keys():
            original_graph.add_node(node, score=node_scores[node])
        
        # Add edges with dummy attributes
        for edge, score in edge_scores.items():
            if edge[0] in original_graph and edge[1] in original_graph:
                original_graph.add_edge(edge[0], edge[1], score=score)
        
        return original_graph
    
    def optimize_connectivity(self, 
                            graph: nx.Graph, 
                            seed_nodes: List[Any]) -> nx.Graph:
        """
        Optimize subgraph for better connectivity between important nodes.
        
        Args:
            graph: NetworkX graph to optimize
            seed_nodes: Important nodes to connect well
            
        Returns:
            nx.Graph: Optimized subgraph
        """
        # Create a copy of the graph
        optimized_graph = graph.copy()
        
        # Filter to seed nodes present in the graph
        present_seeds = [node for node in seed_nodes if node in optimized_graph.nodes]
        
        if len(present_seeds) <= 1:
            # No need for connectivity optimization
            return optimized_graph
        
        # Compute all-pairs shortest paths
        # Dictionary needs to support both int and float (infinity) values
        path_lengths: Dict[Tuple[Any, Any], Union[int, float]] = {}
        for i, source in enumerate(present_seeds):
            for target in present_seeds[i+1:]:
                try:
                    # Find shortest path
                    path = nx.shortest_path(optimized_graph, source=source, target=target)
                    # Explicitly specify this is an int for the number of edges
                    path_lengths[(source, target)] = len(path) - 1  # Number of edges
                except nx.NetworkXNoPath:
                    # No path exists - use float for infinity
                    # This is intentionally a float type as it represents infinity
                    path_lengths[(source, target)] = float('inf')
        
        # Identify pairs with poor connectivity
        poor_connections = [(s, t) for (s, t), length in path_lengths.items() 
                          if length > 3 and length < float('inf')]
        
        # Sort by path length (descending)
        poor_connections.sort(key=lambda pair: path_lengths[pair], reverse=True)
        
        # Improve connectivity for worst connections
        for source, target in poor_connections:
            # Find shortest path
            path = nx.shortest_path(optimized_graph, source=source, target=target)
            
            # If path is still too long, add a direct edge
            if len(path) > 4:  # More than 3 edges
                optimized_graph.add_edge(
                    source, target, weight=1.0, optimized=True)
                logger.info(f"Added direct connection between seed nodes {source} and {target}")
        
        return optimized_graph
    
    def optimize_diversity(self, 
                         graph: nx.Graph, 
                         node_attrs: List[str],
                         edge_attrs: List[str],
                         seed_nodes: List[Any],
                         diversity_measure: str = 'attribute') -> nx.Graph:
        """
        Optimize subgraph for diversity of attributes and content.
        
        Args:
            graph: NetworkX graph to optimize
            node_attrs: Node attributes to consider for diversity
            edge_attrs: Edge attributes to consider for diversity
            seed_nodes: Important nodes to preserve
            diversity_measure: Method for measuring diversity
            
        Returns:
            nx.Graph: Optimized subgraph
        """
        # Create a copy of the graph
        optimized_graph = graph.copy()
        
        # Calculate initial diversity
        initial_diversity = self._calculate_diversity(
            optimized_graph, node_attrs, edge_attrs, diversity_measure)
        
        # Define nodes that can be substituted (not seed nodes)
        swappable_nodes = [node for node in optimized_graph.nodes 
                          if node not in seed_nodes]
        
        # Get initial attribute distributions
        # Dictionary to track attribute value occurrences
        attr_counts: Dict[str, Dict[Any, int]] = {}
        for attr in node_attrs:
            attr_counts[attr] = {}
            for node in optimized_graph.nodes:
                value = optimized_graph.nodes[node].get(attr)
                if value is not None:
                    attr_counts[attr][value] = attr_counts[attr].get(value, 0) + 1
        
        # Identify nodes with most common attribute values
        redundant_nodes = []
        for node in swappable_nodes:
            redundancy_score = 0
            for attr in node_attrs:
                value = optimized_graph.nodes[node].get(attr)
                if value is not None and attr_counts[attr].get(value, 0) > 1:
                    redundancy_score += attr_counts[attr][value]
            
            redundant_nodes.append((node, redundancy_score))
        
        # Sort by redundancy (descending)
        redundant_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Try to replace most redundant nodes with diverse ones
        for node, _ in redundant_nodes[:10]:  # Limit to top 10 most redundant
            # Get neighbors of this node
            neighbors = list(optimized_graph.neighbors(node))
            
            # Find candidate replacements from neighbors of neighbors
            candidates = set()
            for neighbor in neighbors:
                for nn in optimized_graph.neighbors(neighbor):
                    if nn not in optimized_graph and nn not in candidates:
                        candidates.add(nn)
            
            # Evaluate each candidate for diversity improvement
            best_diversity = initial_diversity
            best_candidate = None
            
            for candidate in candidates:
                # Create temporary graph with candidate replacing node
                temp_graph = optimized_graph.copy()
                temp_graph.remove_node(node)
                
                # Add candidate with its attributes
                if candidate in graph:
                    temp_graph.add_node(candidate, **graph.nodes[candidate])
                    
                    # Connect to original neighbors
                    for neighbor in neighbors:
                        if neighbor in temp_graph:
                            if graph.has_edge(candidate, neighbor):
                                temp_graph.add_edge(
                                    candidate, neighbor, **graph.edges[candidate, neighbor])
                            else:
                                temp_graph.add_edge(candidate, neighbor)
                
                # Calculate new diversity
                new_diversity = self._calculate_diversity(
                    temp_graph, node_attrs, edge_attrs, diversity_measure)
                
                # Check if this improves diversity
                if new_diversity > best_diversity:
                    best_diversity = new_diversity
                    best_candidate = candidate
            
            # Apply best substitution if found
            if best_candidate is not None:
                optimized_graph.remove_node(node)
                optimized_graph.add_node(best_candidate, **graph.nodes[best_candidate])
                
                # Connect to original neighbors
                for neighbor in neighbors:
                    if neighbor in optimized_graph:
                        if graph.has_edge(best_candidate, neighbor):
                            optimized_graph.add_edge(
                                best_candidate, neighbor, **graph.edges[best_candidate, neighbor])
                        else:
                            optimized_graph.add_edge(best_candidate, neighbor)
                
                # Update diversity
                initial_diversity = best_diversity
                logger.info(f"Improved diversity by replacing node {node} with {best_candidate}")
        
        return optimized_graph
    
    def _calculate_diversity(self, 
                          graph: nx.Graph, 
                          node_attrs: List[str],
                          edge_attrs: List[str],
                          diversity_measure: str) -> float:
        """
        Calculate diversity score for a graph.
        
        Args:
            graph: NetworkX graph to calculate score for
            node_attrs: Node attributes to consider for diversity
            edge_attrs: Edge attributes to consider for diversity
            diversity_measure: Method for measuring diversity
            
        Returns:
            float: Diversity score
        """
        if diversity_measure == 'attribute':
            # Calculate attribute diversity using entropy
            entropy_sum = 0.0
            
            # Node attribute entropy
            for attr in node_attrs:
                # Count attribute values
                value_counts: Dict[Any, int] = {}
                total = 0
                
                for node in graph.nodes:
                    value = graph.nodes[node].get(attr)
                    if value is not None:
                        value_counts[value] = value_counts.get(value, 0) + 1
                        total += 1
                
                # Calculate entropy
                if total > 0:
                    entropy = 0.0
                    for count in value_counts.values():
                        p = count / total
                        entropy -= p * math.log2(p)
                    
                    # Normalize by max possible entropy
                    max_entropy = math.log2(len(value_counts)) if len(value_counts) > 0 else 0
                    if max_entropy > 0:
                        normalized_entropy = entropy / max_entropy
                    else:
                        normalized_entropy = 0
                        
                    entropy_sum += normalized_entropy
            
            # Edge attribute entropy (similar calculation)
            for attr in edge_attrs:
                value_counts = {}
                total = 0
                
                for _, _, data in graph.edges(data=True):
                    value = data.get(attr)
                    if value is not None:
                        value_counts[value] = value_counts.get(value, 0) + 1
                        total += 1
                
                if total > 0:
                    entropy = 0.0
                    for count in value_counts.values():
                        p = count / total
                        entropy -= p * math.log2(p)
                    
                    max_entropy = math.log2(len(value_counts)) if len(value_counts) > 0 else 0
                    if max_entropy > 0:
                        normalized_entropy = entropy / max_entropy
                    else:
                        normalized_entropy = 0
                        
                    entropy_sum += normalized_entropy
            
            # Average entropy across all attributes
            total_attrs = len(node_attrs) + len(edge_attrs)
            if total_attrs > 0:
                return entropy_sum / total_attrs
            else:
                return 0.0
        
        elif diversity_measure == 'structural':
            # Assess structural diversity
            # Use metrics like clustering coefficient distribution, degree distribution, etc.
            
            # Get degree distribution entropy using a safe approach
            # Collect degree values safely for all nodes
            degree_values: List[int] = []
            
            # Get degree for each node individually to avoid any iteration issues
            for n in graph.nodes():
                try:
                    # Try to get degree directly
                    deg = graph.degree(n)
                    # Ensure it's an int
                    if isinstance(deg, int):
                        degree_values.append(deg)
                    else:
                        # Default if not an int
                        degree_values.append(1)
                except (TypeError, ValueError, AttributeError):
                    # Default to 1 if we can't get the degree
                    degree_values.append(1)
                    
            # Dictionary to count occurrences of each degree value
            degree_counts: Dict[int, int] = {}
            total_nodes = graph.number_of_nodes()
            
            # Count frequencies of each degree value
            for d in degree_values:
                # Ensure the key is an int
                d_int = int(d)
                degree_counts[d_int] = degree_counts.get(d_int, 0) + 1
            
            # Calculate entropy
            entropy = 0.0
            for count in degree_counts.values():
                p = count / total_nodes
                entropy -= p * math.log2(p)
            
            # Normalize
            max_entropy = math.log2(len(degree_counts)) if len(degree_counts) > 0 else 0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            return normalized_entropy
        
        else:
            # Default measure
            logger.warning(f"Unknown diversity measure: {diversity_measure}, using attribute")
            return self._calculate_diversity(graph, node_attrs, edge_attrs, 'attribute')
    
    def optimize_hybrid(self, 
                      graph: nx.Graph, 
                      node_scores: Dict[Any, float],
                      edge_scores: Dict[Tuple[Any, Any], float],
                      seed_nodes: List[Any],
                      node_attrs: List[str] = ['type', 'category'],
                      edge_attrs: List[str] = ['type', 'relation']) -> nx.Graph:
        """
        Apply hybrid optimization for relevance, connectivity, and diversity.
        
        Args:
            graph: NetworkX graph to optimize
            node_scores: Dictionary of node -> relevance score
            edge_scores: Dictionary of edge -> relevance score
            seed_nodes: Important nodes to preserve
            node_attrs: Node attributes to consider for diversity
            edge_attrs: Edge attributes to consider for diversity
            
        Returns:
            nx.Graph: Optimized subgraph
        """
        # Create a copy of the graph
        optimized_graph = graph.copy()
        
        # Step 1: Optimize for relevance
        relevance_graph = self.optimize_relevance(
            optimized_graph, node_scores, edge_scores, seed_nodes)
        
        # Step 2: Optimize for connectivity
        connectivity_graph = self.optimize_connectivity(
            relevance_graph, seed_nodes)
        
        # Step 3: Optimize for diversity
        diversity_graph = self.optimize_diversity(
            connectivity_graph, node_attrs, edge_attrs, seed_nodes)
        
        # Calculate scores for each dimension
        relevance_score = self._calculate_total_relevance(
            diversity_graph, node_scores, edge_scores)
        
        logger.info(f"Hybrid optimization complete with final relevance score: {relevance_score:.4f}")
        
        return diversity_graph
