"""
Graph Optimizer for GraphRAG

This module implements the GraphOptimizer class that optimizes graph structures
for retrieval operations in the GraphRAG implementation.
"""

import networkx as nx
from typing import Dict, List, Optional, Any, Set, Tuple, Union
import logging
import numpy as np
from datetime import datetime
import os

from ..models.entity import Entity
from ..models.relationship import Relationship


class GraphOptimizer:
    """
    Class for optimizing graph structures for retrieval operations.
    
    The GraphOptimizer provides methods for improving graph quality,
    reducing noise, and enhancing retrieval performance through
    various graph optimization techniques.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the graph optimizer.
        
        Args:
            config: Configuration dictionary with the following options:
                - min_edge_weight: Minimum weight for edges to keep
                - min_node_degree: Minimum degree for nodes to keep
                - max_node_degree: Maximum degree for nodes (prune excess edges)
                - prune_isolated_nodes: Whether to remove isolated nodes
                - compute_centrality: Whether to compute centrality measures
                - compute_community: Whether to detect communities
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Get configuration options
        self.min_edge_weight = self.config.get("min_edge_weight", 0.1)
        self.min_node_degree = self.config.get("min_node_degree", 1)
        self.max_node_degree = self.config.get("max_node_degree", 100)
        self.prune_isolated_nodes = self.config.get("prune_isolated_nodes", True)
        self.compute_centrality = self.config.get("compute_centrality", True)
        self.compute_community = self.config.get("compute_community", True)
        
        self.logger.info(f"Initialized GraphOptimizer with min_edge_weight={self.min_edge_weight}, "
                        f"min_node_degree={self.min_node_degree}")
    
    def optimize_graph(self, graph: nx.Graph) -> nx.Graph:
        """
        Optimize a graph for retrieval operations.
        
        Args:
            graph: NetworkX graph to optimize
            
        Returns:
            Optimized graph
        """
        self.logger.info(f"Optimizing graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        # Create a copy of the graph
        optimized_graph = graph.copy()
        
        # Perform optimization steps
        optimized_graph = self.prune_low_weight_edges(optimized_graph)
        optimized_graph = self.prune_high_degree_nodes(optimized_graph)
        
        if self.prune_isolated_nodes:
            optimized_graph = self.remove_isolated_nodes(optimized_graph)
            
        if self.compute_centrality:
            optimized_graph = self.add_centrality_measures(optimized_graph)
            
        if self.compute_community:
            optimized_graph = self.detect_communities(optimized_graph)
            
        # Add optimization metadata
        optimized_graph.graph["optimized_at"] = datetime.now().isoformat()
        optimized_graph.graph["optimization_config"] = {
            "min_edge_weight": self.min_edge_weight,
            "min_node_degree": self.min_node_degree,
            "max_node_degree": self.max_node_degree,
            "prune_isolated_nodes": self.prune_isolated_nodes,
            "compute_centrality": self.compute_centrality,
            "compute_community": self.compute_community
        }
        
        self.logger.info(f"Optimized graph has {optimized_graph.number_of_nodes()} nodes and {optimized_graph.number_of_edges()} edges")
        return optimized_graph
    
    def prune_low_weight_edges(self, graph: nx.Graph) -> nx.Graph:
        """
        Remove edges with weight below the minimum threshold.
        
        Args:
            graph: NetworkX graph to prune
            
        Returns:
            Pruned graph
        """
        edges_to_remove = []
        
        for u, v, attrs in graph.edges(data=True):
            weight = attrs.get("weight", 1.0)
            
            if weight < self.min_edge_weight:
                edges_to_remove.append((u, v))
                
        graph.remove_edges_from(edges_to_remove)
        self.logger.info(f"Removed {len(edges_to_remove)} low-weight edges")
        
        return graph
    
    def prune_high_degree_nodes(self, graph: nx.Graph) -> nx.Graph:
        """
        Prune excess edges from high-degree nodes.
        
        For nodes with degree > max_node_degree, keep only the
        max_node_degree highest-weight edges.
        
        Args:
            graph: NetworkX graph to prune
            
        Returns:
            Pruned graph
        """
        if self.max_node_degree <= 0:
            return graph
            
        edges_to_remove = []
        
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            degree = len(neighbors)
            
            if degree > self.max_node_degree:
                # Get all edges with weights
                weighted_edges = []
                for neighbor in neighbors:
                    weight = graph[node][neighbor].get("weight", 1.0)
                    weighted_edges.append((neighbor, weight))
                    
                # Sort by weight (descending)
                weighted_edges.sort(key=lambda x: x[1], reverse=True)
                
                # Keep only the top max_node_degree edges
                edges_to_keep = weighted_edges[:self.max_node_degree]
                keep_neighbors = {neighbor for neighbor, _ in edges_to_keep}
                
                # Mark excess edges for removal
                for neighbor in neighbors:
                    if neighbor not in keep_neighbors:
                        edges_to_remove.append((node, neighbor))
                        
        graph.remove_edges_from(edges_to_remove)
        self.logger.info(f"Removed {len(edges_to_remove)} edges from high-degree nodes")
        
        return graph
    
    def remove_isolated_nodes(self, graph: nx.Graph) -> nx.Graph:
        """
        Remove isolated nodes (degree < min_node_degree).
        
        Args:
            graph: NetworkX graph to prune
            
        Returns:
            Pruned graph
        """
        nodes_to_remove = []
        
        for node, degree in graph.degree():
            if degree < self.min_node_degree:
                # Don't remove document nodes
                if graph.nodes[node].get("node_type") != "document":
                    nodes_to_remove.append(node)
                    
        graph.remove_nodes_from(nodes_to_remove)
        self.logger.info(f"Removed {len(nodes_to_remove)} isolated nodes")
        
        return graph
    
    def add_centrality_measures(self, graph: nx.Graph) -> nx.Graph:
        """
        Compute and add centrality measures to the graph.
        
        Args:
            graph: NetworkX graph to enhance
            
        Returns:
            Enhanced graph with centrality measures
        """
        self.logger.info("Computing centrality measures")
        
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(graph)
            nx.set_node_attributes(graph, degree_centrality, "degree_centrality")
            
            # Betweenness centrality (sample for large graphs)
            if graph.number_of_nodes() > 1000:
                k = min(500, graph.number_of_nodes())
                betweenness = nx.betweenness_centrality(graph, k=k)
            else:
                betweenness = nx.betweenness_centrality(graph)
            nx.set_node_attributes(graph, betweenness, "betweenness_centrality")
            
            # Closeness centrality (sample for large graphs)
            if graph.number_of_nodes() > 1000:
                k = min(500, graph.number_of_nodes())
                closeness = nx.closeness_centrality(graph, nbunch=np.random.choice(
                    list(graph.nodes()), k, replace=False).tolist())
            else:
                closeness = nx.closeness_centrality(graph)
            nx.set_node_attributes(graph, closeness, "closeness_centrality")
            
            # PageRank
            pagerank = nx.pagerank(graph, weight="weight")
            nx.set_node_attributes(graph, pagerank, "pagerank")
            
            self.logger.info("Added centrality measures to graph")
        except Exception as e:
            self.logger.warning(f"Failed to compute some centrality measures: {str(e)}")
            
        return graph
    
    def detect_communities(self, graph: nx.Graph) -> nx.Graph:
        """
        Detect communities in the graph and add community labels.
        
        Args:
            graph: NetworkX graph to enhance
            
        Returns:
            Enhanced graph with community labels
        """
        self.logger.info("Detecting communities")
        
        try:
            # Use Louvain community detection
            from community import best_partition
            
            # Convert to undirected if directed
            if graph.is_directed():
                undirected_graph = graph.to_undirected()
                partition = best_partition(undirected_graph)
            else:
                partition = best_partition(graph)
                
            # Add community labels to nodes
            nx.set_node_attributes(graph, partition, "community")
            
            # Count communities
            communities = set(partition.values())
            self.logger.info(f"Detected {len(communities)} communities")
            
            # Add community size information
            community_sizes = {}
            for community_id in communities:
                size = sum(1 for v in partition.values() if v == community_id)
                community_sizes[community_id] = size
                
            graph.graph["community_sizes"] = community_sizes
            
        except ImportError:
            self.logger.warning("Community detection requires the python-louvain package")
        except Exception as e:
            self.logger.warning(f"Failed to detect communities: {str(e)}")
            
        return graph
    
    def optimize_for_query(self, graph: nx.Graph, query_entities: List[Entity]) -> nx.Graph:
        """
        Optimize a graph specifically for a query with given entities.
        
        This creates a query-specific subgraph that focuses on the relevant
        parts of the graph for the given query entities.
        
        Args:
            graph: NetworkX graph to optimize
            query_entities: List of entities from the query
            
        Returns:
            Query-optimized subgraph
        """
        self.logger.info(f"Optimizing graph for query with {len(query_entities)} entities")
        
        # Find entity nodes in the graph
        entity_nodes = []
        for entity in query_entities:
            # Try exact match first
            if graph.has_node(entity.id):
                entity_nodes.append(entity.id)
                continue
                
            # Try text match
            entity_text = entity.text.lower()
            for node, attrs in graph.nodes(data=True):
                node_text = attrs.get("label", "").lower()
                if node_text == entity_text:
                    entity_nodes.append(node)
                    break
                    
        self.logger.info(f"Found {len(entity_nodes)} matching nodes for query entities")
        
        if not entity_nodes:
            self.logger.warning("No matching nodes found for query entities")
            return graph.copy()
            
        # Create a subgraph centered on the query entities
        max_distance = self.config.get("query_max_distance", 2)
        nodes_to_include = set(entity_nodes)
        
        # Add neighbors within max_distance
        for entity_node in entity_nodes:
            try:
                for node, distance in nx.single_source_shortest_path_length(
                    graph, entity_node, cutoff=max_distance).items():
                    nodes_to_include.add(node)
            except nx.NetworkXError:
                continue
                
        # Create the subgraph
        query_graph = graph.subgraph(nodes_to_include).copy()
        
        # Add query metadata
        query_graph.graph["query_optimized"] = True
        query_graph.graph["query_entities"] = [entity.text for entity in query_entities]
        query_graph.graph["optimized_at"] = datetime.now().isoformat()
        
        self.logger.info(f"Created query-optimized subgraph with {query_graph.number_of_nodes()} nodes "
                        f"and {query_graph.number_of_edges()} edges")
        
        return query_graph
    
    def compute_node_importance(self, graph: nx.Graph) -> Dict[str, float]:
        """
        Compute importance scores for all nodes in the graph.
        
        This combines multiple centrality measures to create a single
        importance score for each node.
        
        Args:
            graph: NetworkX graph to analyze
            
        Returns:
            Dictionary mapping node IDs to importance scores
        """
        self.logger.info("Computing node importance scores")
        
        # Ensure centrality measures are computed
        if self.compute_centrality and not any("centrality" in attr for _, attr in graph.nodes(data=True)):
            graph = self.add_centrality_measures(graph)
            
        # Compute importance scores
        importance_scores = {}
        
        for node, attrs in graph.nodes(data=True):
            # Combine multiple centrality measures
            degree_cent = attrs.get("degree_centrality", 0.0)
            betweenness_cent = attrs.get("betweenness_centrality", 0.0)
            closeness_cent = attrs.get("closeness_centrality", 0.0)
            pagerank = attrs.get("pagerank", 0.0)
            
            # Weighted combination
            importance = (
                0.3 * degree_cent +
                0.3 * betweenness_cent +
                0.2 * closeness_cent +
                0.2 * pagerank
            )
            
            importance_scores[node] = importance
            
        # Normalize scores
        if importance_scores:
            max_score = max(importance_scores.values())
            if max_score > 0:
                for node in importance_scores:
                    importance_scores[node] /= max_score
                    
        # Add to graph
        nx.set_node_attributes(graph, importance_scores, "importance")
        
        return importance_scores
    
    def prune_redundant_paths(self, graph: nx.Graph) -> nx.Graph:
        """
        Prune redundant paths between nodes.
        
        For pairs of nodes with multiple paths, keep only the strongest paths.
        
        Args:
            graph: NetworkX graph to optimize
            
        Returns:
            Optimized graph with redundant paths removed
        """
        self.logger.info("Pruning redundant paths")
        
        # Create a copy of the graph
        pruned_graph = graph.copy()
        
        # Find node pairs with multiple paths
        redundant_edges = []
        processed_pairs = set()
        
        for u in pruned_graph.nodes():
            for v in pruned_graph.nodes():
                if u == v or (u, v) in processed_pairs or (v, u) in processed_pairs:
                    continue
                    
                # Check if there are multiple paths
                try:
                    paths = list(nx.all_simple_paths(pruned_graph, u, v, cutoff=3))
                    if len(paths) > 1:
                        # Calculate path strengths
                        path_strengths = []
                        for path in paths:
                            # Calculate path strength as product of edge weights
                            strength = 1.0
                            for i in range(len(path) - 1):
                                weight = pruned_graph[path[i]][path[i+1]].get("weight", 1.0)
                                strength *= weight
                                
                            path_strengths.append((path, strength))
                            
                        # Sort by strength (descending)
                        path_strengths.sort(key=lambda x: x[1], reverse=True)
                        
                        # Keep the strongest path, mark others for removal
                        strongest_path = path_strengths[0][0]
                        for path, _ in path_strengths[1:]:
                            for i in range(len(path) - 1):
                                # Check if this edge is not in the strongest path
                                edge_in_strongest = False
                                for j in range(len(strongest_path) - 1):
                                    if (path[i] == strongest_path[j] and 
                                        path[i+1] == strongest_path[j+1]):
                                        edge_in_strongest = True
                                        break
                                        
                                if not edge_in_strongest:
                                    redundant_edges.append((path[i], path[i+1]))
                                    
                except nx.NetworkXNoPath:
                    continue
                    
                processed_pairs.add((u, v))
                
        # Remove redundant edges
        pruned_graph.remove_edges_from(redundant_edges)
        self.logger.info(f"Removed {len(redundant_edges)} redundant edges")
        
        return pruned_graph
