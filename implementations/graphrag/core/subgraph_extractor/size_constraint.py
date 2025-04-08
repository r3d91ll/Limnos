"""
Size-Constrained Subgraph Extraction.

This module provides functionality for extracting subgraphs under various
size constraints to manage complexity and relevance.
"""

import logging
import heapq
from typing import Dict, List, Set, Any, Optional, Callable, Tuple, Union
import networkx as nx
import numpy as np

from .relevance_scorer import NodeEdgeScorer

logger = logging.getLogger(__name__)

class SizeConstrainer:
    """
    Provides methods for extracting subgraphs under size constraints.
    """
    
    def __init__(self, 
                max_nodes: int = 100,
                max_edges: int = 500,
                max_density: float = 0.1,
                prioritize_by: str = 'relevance',
                preserve_seed_nodes: bool = True,
                balance_threshold: float = 0.5):
        """
        Initialize the size constrainer with configurable parameters.
        
        Args:
            max_nodes: Maximum number of nodes in extracted subgraph
            max_edges: Maximum number of edges in extracted subgraph
            max_density: Maximum edge density (edges/max_possible_edges)
            prioritize_by: Method for prioritizing nodes ('relevance', 'degree', 'connectivity')
            preserve_seed_nodes: Whether to always keep seed nodes
            balance_threshold: Balance between relevance and connectivity (0-1)
                               1 = more relevance, 0 = more connectivity
        """
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_density = max_density
        self.prioritize_by = prioritize_by
        self.preserve_seed_nodes = preserve_seed_nodes
        self.balance_threshold = balance_threshold
        
        logger.info(f"Initialized SizeConstrainer with max_nodes={max_nodes}, "
                   f"max_edges={max_edges}, prioritize_by={prioritize_by}")
    
    def constrain_by_size(self, 
                        graph: nx.Graph, 
                        seed_nodes: List[Any],
                        node_scores: Optional[Dict[Any, float]] = None) -> nx.Graph:
        """
        Extract a subgraph constrained by node and edge limits.
        
        Args:
            graph: NetworkX graph to constrain
            seed_nodes: Important nodes to preserve
            node_scores: Dictionary of node -> relevance score
            
        Returns:
            nx.Graph: Size-constrained subgraph
        """
        # If graph already meets constraints, return it
        if (graph.number_of_nodes() <= self.max_nodes and 
            graph.number_of_edges() <= self.max_edges):
            return graph
        
        # Create a new graph with seed nodes
        constrained_graph: nx.Graph = nx.Graph()
        
        # Add seed nodes
        for node in seed_nodes:
            if node in graph:
                constrained_graph.add_node(node, **graph.nodes[node])
        
        # Initialize priority queue (list of tuples with priority and node)
        priority_queue: List[Tuple[float, Any]] = []
        
        # Initial set of visited nodes (seed nodes)
        visited = set(seed_nodes)
        
        # Initialize frontier from neighbors of seed nodes
        for seed in seed_nodes:
            if seed not in graph:
                continue
                
            for neighbor in graph.neighbors(seed):
                if neighbor not in visited:
                    # Calculate priority
                    priority = self._calculate_node_priority(
                        graph, neighbor, seed_nodes, node_scores)
                    
                    # Add to priority queue (negative priority for max-heap)
                    heapq.heappush(priority_queue, (-priority, neighbor))
        
        # Add nodes until max_nodes is reached or queue is empty
        while (len(constrained_graph) < self.max_nodes and 
              len(priority_queue) > 0):
            # Get highest priority node
            _, node = heapq.heappop(priority_queue)
            
            # Skip if already added
            if node in constrained_graph:
                continue
            
            # Add node to graph
            constrained_graph.add_node(node, **graph.nodes[node])
            visited.add(node)
            
            # Add edges to existing nodes in constrained graph
            for neighbor in graph.neighbors(node):
                if neighbor in constrained_graph:
                    # Check edge limit
                    if constrained_graph.number_of_edges() < self.max_edges:
                        # Add edge with attributes
                        constrained_graph.add_edge(
                            node, neighbor, **graph.edges[node, neighbor])
            
            # Add neighbors to priority queue
            for neighbor in graph.neighbors(node):
                if (neighbor not in visited and 
                    neighbor not in [n for _, n in priority_queue]):
                    # Calculate priority
                    priority = self._calculate_node_priority(
                        graph, neighbor, seed_nodes, node_scores)
                    
                    # Add to priority queue
                    heapq.heappush(priority_queue, (-priority, neighbor))
        
        logger.info(f"Constrained graph from {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges "
                   f"to {constrained_graph.number_of_nodes()} nodes and {constrained_graph.number_of_edges()} edges")
                   
        return constrained_graph
    
    def _calculate_node_priority(self, 
                               graph: nx.Graph, 
                               node: Any, 
                               seed_nodes: List[Any],
                               node_scores: Optional[Dict[Any, float]] = None) -> float:
        """
        Calculate priority score for a node based on the prioritization method.
        
        Args:
            graph: Original graph
            node: Node to calculate priority for
            seed_nodes: Seed nodes in the graph
            node_scores: Dictionary of node -> relevance score
            
        Returns:
            float: Priority score
        """
        # Base priority components
        relevance_score = 0.0
        connectivity_score = 0.0
        degree_score = 0.0
        
        # Relevance component (if scores provided)
        if node_scores and node in node_scores:
            relevance_score = node_scores[node]
        
        # Connectivity component (connections to seed nodes)
        seed_connections = sum(1 for seed in seed_nodes if graph.has_edge(node, seed))
        if seed_nodes:
            connectivity_score = seed_connections / len(seed_nodes)
        
        # Degree component (normalized by max degree)
        # Safe way to get degree information that works with all NetworkX versions
        degree_values = []
        
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
            
        # Get degree for this specific node safely
        try:
            # Try to get degree directly for this node
            node_deg = graph.degree(node)
            # Ensure we have an int
            node_degree = node_deg if isinstance(node_deg, int) else 0
        except (TypeError, ValueError, AttributeError):
            # Default to 0 if we can't get the degree
            node_degree = 0
            
        # Calculate normalized degree score
        degree_score = float(node_degree) / float(max_degree)
        
        # Calculate final priority based on method
        if self.prioritize_by == 'relevance':
            # Weighted combination of relevance and connectivity
            priority = (self.balance_threshold * relevance_score + 
                       (1 - self.balance_threshold) * connectivity_score)
        
        elif self.prioritize_by == 'degree':
            # Weighted combination of degree and connectivity
            priority = (self.balance_threshold * degree_score + 
                       (1 - self.balance_threshold) * connectivity_score)
        
        elif self.prioritize_by == 'connectivity':
            # Pure connectivity to seed nodes
            priority = connectivity_score
        
        else:
            # Default to balanced approach
            priority = (relevance_score + connectivity_score + degree_score) / 3
        
        return priority
    
    def constrain_by_density(self, 
                           graph: nx.Graph, 
                           node_scores: Dict[Any, float],
                           seed_nodes: Optional[List[Any]] = None) -> nx.Graph:
        """
        Extract a subgraph constrained by edge density.
        
        Args:
            graph: NetworkX graph to constrain
            node_scores: Dictionary of node -> relevance score
            seed_nodes: Important nodes to preserve
            
        Returns:
            nx.Graph: Density-constrained subgraph
        """
        # If graph already meets density constraint, return it
        if graph.number_of_nodes() <= 1:
            return graph
            
        current_density = (graph.number_of_edges() / 
                          (graph.number_of_nodes() * (graph.number_of_nodes() - 1) / 2))
                          
        if current_density <= self.max_density:
            return graph
        
        # Create a copy of the graph
        constrained_graph = graph.copy()
        
        # Sort nodes by score (ascending)
        sorted_nodes = sorted(
            node_scores.items(), key=lambda x: x[1])
        
        # Remove nodes until we reach desired density, starting with lowest score
        for node, _ in sorted_nodes:
            # Skip seed nodes if preserve_seed_nodes is True
            if self.preserve_seed_nodes and seed_nodes and node in seed_nodes:
                continue
            
            # Calculate current density
            n = constrained_graph.number_of_nodes()
            m = constrained_graph.number_of_edges()
            
            if n <= 1:
                # Can't reduce further
                break
                
            current_density = m / (n * (n - 1) / 2)
            
            if current_density <= self.max_density:
                # Reached target density
                break
            
            # Remove node
            constrained_graph.remove_node(node)
        
        # Calculate final density
        n = constrained_graph.number_of_nodes()
        m = constrained_graph.number_of_edges()
        
        if n > 1:
            final_density = m / (n * (n - 1) / 2)
            logger.info(f"Constrained graph density from {current_density:.4f} to {final_density:.4f}")
        else:
            logger.info("Constrained graph contains only 1 or 0 nodes")
        
        return constrained_graph
    
    def ensure_connectedness(self, 
                           graph: nx.Graph, 
                           seed_nodes: Optional[List[Any]] = None) -> nx.Graph:
        """
        Ensure the subgraph is connected, keeping the largest component containing seed nodes.
        
        Args:
            graph: NetworkX graph to process
            seed_nodes: Important nodes to preserve
            
        Returns:
            nx.Graph: Connected subgraph
        """
        # Check if already connected
        if nx.is_connected(graph):
            return graph
        
        # Find connected components
        components = list(nx.connected_components(graph))
        
        if len(components) == 1:
            return graph
        
        # If no seed nodes, just keep the largest component
        if not seed_nodes:
            largest_component = max(components, key=len)
            return graph.subgraph(largest_component).copy()
        
        # Find components containing seed nodes
        seed_components = []
        for component in components:
            if any(seed in component for seed in seed_nodes):
                seed_components.append(component)
        
        # If no component contains seed nodes, keep the largest component
        if not seed_components:
            largest_component = max(components, key=len)
            return graph.subgraph(largest_component).copy()
        
        # If multiple components contain seed nodes, keep the largest one
        largest_seed_component = max(seed_components, key=len)
        
        # Create a new graph with just this component
        connected_graph = graph.subgraph(largest_seed_component).copy()
        
        # Add any missing seed nodes
        for seed in seed_nodes:
            if seed in graph and seed not in connected_graph:
                connected_graph.add_node(seed, **graph.nodes[seed])
                
                # Find closest node in the component to connect to
                min_distance = float('inf')
                closest_node = None
                
                # This is inefficient for large graphs but ensures connectivity
                for node in largest_seed_component:
                    try:
                        # Try to find a path in the original graph
                        path = nx.shortest_path(graph, seed, node)
                        if len(path) < min_distance:
                            min_distance = len(path)
                            closest_node = node
                    except nx.NetworkXNoPath:
                        continue
                
                # Connect to closest node if found
                if closest_node is not None:
                    connected_graph.add_edge(seed, closest_node, weight=1.0, preserved=True)
        
        logger.info(f"Ensured connectedness, keeping {connected_graph.number_of_nodes()} nodes "
                   f"out of {graph.number_of_nodes()} original nodes")
                   
        return connected_graph
