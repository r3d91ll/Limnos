"""
Context-Aware Pruning for Knowledge Graphs.

This module provides functionality for pruning knowledge graphs based on context
relevance to maintain the most informative parts of the graph for a given query.
"""

import logging
import math
from typing import Dict, List, Set, Any, Optional, Callable, Tuple, Union
import networkx as nx
import numpy as np

from .relevance_scorer import NodeEdgeScorer

logger = logging.getLogger(__name__)

class ContextPruner:
    """
    Provides context-aware pruning algorithms for knowledge graphs.
    """
    
    def __init__(self, 
                node_threshold: float = 0.3,
                edge_threshold: float = 0.2,
                preserve_connectivity: bool = True,
                preserve_seed_nodes: bool = True,
                preserve_edge_weight_attr: Optional[str] = 'weight',
                max_edge_distance: int = 3):
        """
        Initialize the context pruner with configurable parameters.
        
        Args:
            node_threshold: Threshold for node relevance score (nodes below are pruned)
            edge_threshold: Threshold for edge relevance score (edges below are pruned)
            preserve_connectivity: Whether to preserve connectivity in pruned graph
            preserve_seed_nodes: Whether to always keep seed nodes
            preserve_edge_weight_attr: Attribute name for edge weights
            max_edge_distance: Maximum distance for preserving connectivity
        """
        self.node_threshold = node_threshold
        self.edge_threshold = edge_threshold
        self.preserve_connectivity = preserve_connectivity
        self.preserve_seed_nodes = preserve_seed_nodes
        self.preserve_edge_weight_attr = preserve_edge_weight_attr
        self.max_edge_distance = max_edge_distance
        
        logger.info(f"Initialized ContextPruner with node_threshold={node_threshold}, "
                   f"edge_threshold={edge_threshold}")
    
    def prune_by_relevance(self, 
                         graph: nx.Graph, 
                         node_scores: Dict[Any, float],
                         edge_scores: Dict[Tuple[Any, Any], float],
                         seed_nodes: Optional[List[Any]] = None) -> nx.Graph:
        """
        Prune a graph based on node and edge relevance scores.
        
        Args:
            graph: NetworkX graph to prune
            node_scores: Dictionary of node -> relevance score
            edge_scores: Dictionary of edge -> relevance score
            seed_nodes: Important nodes to preserve
            
        Returns:
            nx.Graph: Pruned graph
        """
        # Create a copy of the graph to modify
        pruned_graph = graph.copy()
        
        # Collect nodes to remove (below threshold)
        nodes_to_remove = []
        for node, score in node_scores.items():
            if (score < self.node_threshold and 
                (not self.preserve_seed_nodes or node not in (seed_nodes or []))):
                nodes_to_remove.append(node)
        
        # Remove nodes
        pruned_graph.remove_nodes_from(nodes_to_remove)
        
        # Collect edges to remove (below threshold)
        edges_to_remove = []
        for edge, score in edge_scores.items():
            if score < self.edge_threshold:
                # Check if edge endpoints still exist in graph
                if (edge[0] in pruned_graph.nodes and 
                    edge[1] in pruned_graph.nodes):
                    edges_to_remove.append(edge)
        
        # Remove edges
        pruned_graph.remove_edges_from(edges_to_remove)
        
        # Preserve connectivity if requested
        if self.preserve_connectivity and seed_nodes:
            pruned_graph = self._preserve_connectivity(pruned_graph, seed_nodes)
        
        logger.info(f"Pruned graph from {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges "
                   f"to {pruned_graph.number_of_nodes()} nodes and {pruned_graph.number_of_edges()} edges")
                   
        return pruned_graph
    
    def _preserve_connectivity(self, graph: nx.Graph, seed_nodes: List[Any]) -> nx.Graph:
        """
        Ensure connectivity between seed nodes in the pruned graph.
        
        Args:
            graph: Partially pruned graph
            seed_nodes: Nodes to keep connected
            
        Returns:
            nx.Graph: Graph with preserved connectivity
        """
        # Filter to seed nodes present in the graph
        present_seeds = [node for node in seed_nodes if node in graph.nodes]
        
        if len(present_seeds) <= 1:
            # No need to preserve connectivity with 0 or 1 seed node
            return graph
        
        # Ensure the graph is connected between seed nodes
        connected_graph = graph.copy()
        
        # For each pair of seed nodes
        for i, source in enumerate(present_seeds):
            for target in present_seeds[i+1:]:
                # Check if there's a path in the current graph
                try:
                    # If path exists, no action needed
                    nx.shortest_path(connected_graph, source=source, target=target)
                except nx.NetworkXNoPath:
                    # No path, try to find one in the original graph
                    try:
                        # Find shortest path in original (potentially inefficient for large graphs)
                        path = nx.shortest_path(
                            graph.to_undirected(), source=source, target=target, 
                            weight=self.preserve_edge_weight_attr)
                        
                        # Add missing edges along the path
                        for j in range(len(path) - 1):
                            u, v = path[j], path[j+1]
                            if not connected_graph.has_edge(u, v):
                                # Copy edge attributes if the edge existed in original graph
                                if graph.has_edge(u, v):
                                    connected_graph.add_edge(u, v, **graph.edges[u, v])
                                else:
                                    connected_graph.add_edge(u, v, 
                                                           weight=1.0, 
                                                           preserved=True)
                    except (nx.NetworkXNoPath, nx.NetworkXError):
                        # Cannot find path, nodes might be in different components
                        logger.warning(f"Could not find path between seed nodes {source} and {target}")
        
        return connected_graph
    
    def prune_by_distance(self, 
                        graph: nx.Graph, 
                        center_nodes: List[Any],
                        max_distance: int = 3) -> nx.Graph:
        """
        Prune a graph by keeping only nodes within a certain distance of center nodes.
        
        Args:
            graph: NetworkX graph to prune
            center_nodes: Center nodes for distance calculation
            max_distance: Maximum distance from center nodes to keep
            
        Returns:
            nx.Graph: Pruned graph
        """
        # Create a copy of the graph to modify
        pruned_graph = graph.copy()
        
        # Collect nodes within distance from center nodes
        nodes_to_keep = set(center_nodes)
        
        # For each center node, collect nodes within distance
        for center in center_nodes:
            if center not in graph:
                continue
                
            # Use BFS to find nodes within distance
            visited = {center: 0}  # node -> distance
            queue = [(center, 0)]  # (node, distance)
            
            while queue:
                node, distance = queue.pop(0)
                
                if distance < max_distance:
                    # Add neighbors to queue
                    for neighbor in graph.neighbors(node):
                        if neighbor not in visited:
                            visited[neighbor] = distance + 1
                            queue.append((neighbor, distance + 1))
            
            # Add all visited nodes to keep
            nodes_to_keep.update(visited.keys())
        
        # Remove nodes not in nodes_to_keep
        nodes_to_remove = [node for node in graph.nodes if node not in nodes_to_keep]
        pruned_graph.remove_nodes_from(nodes_to_remove)
        
        logger.info(f"Distance-pruned graph from {graph.number_of_nodes()} nodes "
                   f"to {pruned_graph.number_of_nodes()} nodes")
                   
        return pruned_graph
    
    def adaptive_context_pruning(self, 
                              graph: nx.Graph,
                              query_embedding: np.ndarray,
                              seed_nodes: List[Any],
                              scorer: Optional[NodeEdgeScorer] = None,
                              max_nodes: int = 100) -> nx.Graph:
        """
        Adaptively prune a graph based on context relevance.
        
        This method combines relevance scoring and distance-based pruning,
        adapting thresholds to achieve desired graph size.
        
        Args:
            graph: NetworkX graph to prune
            query_embedding: Query embedding for relevance calculation
            seed_nodes: Seed nodes to preserve
            scorer: NodeEdgeScorer instance (created if None)
            max_nodes: Target maximum number of nodes in pruned graph
            
        Returns:
            nx.Graph: Pruned graph
        """
        # Create scorer if not provided
        if scorer is None:
            scorer = NodeEdgeScorer()
        
        # Score nodes and edges
        node_scores, edge_scores = scorer.score_graph(graph, query_embedding)
        
        # Try different node thresholds until we get close to target size
        # Start with the original threshold
        threshold = self.node_threshold
        step = 0.05
        max_tries = 10
        
        pruned_graph = None
        for i in range(max_tries):
            # Prune with current threshold
            pruned_graph = self.prune_by_relevance(
                graph, node_scores, edge_scores, seed_nodes)
            
            # Check if size is in an acceptable range
            current_size = pruned_graph.number_of_nodes()
            
            if current_size <= max_nodes * 1.1 and current_size >= max_nodes * 0.7:
                # Size is good enough
                break
            
            # Adjust threshold based on current size
            if current_size > max_nodes:
                # Too many nodes, increase threshold
                threshold += step
            else:
                # Too few nodes, decrease threshold
                threshold -= step
                
            # Make sure threshold stays in [0,1]
            threshold = max(0.0, min(1.0, threshold))
            
            # Update node threshold for next iteration
            self.node_threshold = threshold
        
        logger.info(f"Adaptive pruning converged with node_threshold={threshold}, "
                   f"final graph size: {pruned_graph.number_of_nodes()} nodes")
        
        # Reset threshold to original
        self.node_threshold = threshold
        
        return pruned_graph
    
    def prune_by_node_type(self, 
                         graph: nx.Graph,
                         type_attr: str = 'type', 
                         include_types: Optional[List[str]] = None,
                         exclude_types: Optional[List[str]] = None) -> nx.Graph:
        """
        Prune a graph by keeping only nodes of specified types.
        
        Args:
            graph: NetworkX graph to prune
            type_attr: Attribute name containing node type
            include_types: Node types to include (None means include all)
            exclude_types: Node types to exclude (takes precedence over include)
            
        Returns:
            nx.Graph: Pruned graph
        """
        # Create a copy of the graph to modify
        pruned_graph = graph.copy()
        
        # Collect nodes to remove
        nodes_to_remove = []
        
        for node in graph.nodes:
            node_type = graph.nodes[node].get(type_attr)
            
            # Check if node should be excluded
            if exclude_types and node_type in exclude_types:
                nodes_to_remove.append(node)
                continue
                
            # Check if node should be included
            if include_types and node_type not in include_types:
                nodes_to_remove.append(node)
        
        # Remove nodes
        pruned_graph.remove_nodes_from(nodes_to_remove)
        
        logger.info(f"Type-pruned graph from {graph.number_of_nodes()} nodes "
                   f"to {pruned_graph.number_of_nodes()} nodes")
                   
        return pruned_graph
