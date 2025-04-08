"""
Main Subgraph Extractor for GraphRAG.

This module provides the main interface for extracting relevant subgraphs
from a knowledge graph based on query context and relevance criteria.
"""

import logging
import math
from typing import Dict, List, Set, Any, Optional, Callable, Tuple, Union
import networkx as nx
import numpy as np

from .relevance_scorer import NodeEdgeScorer
from .context_pruner import ContextPruner
from .size_constraint import SizeConstrainer
from .subgraph_optimizer import SubgraphOptimizer
from .config import (
    SubgraphExtractorConfig, NodeEdgeScorerConfig, ContextPrunerConfig,
    SizeConstrainerConfig, SubgraphOptimizerConfig, ExtractorPresets
)

logger = logging.getLogger(__name__)

class SubgraphExtractor:
    """
    Main class for extracting relevant subgraphs from a knowledge graph.
    
    This class integrates all the subgraph extraction components:
    - Relevance scoring for nodes and edges
    - Context-aware pruning based on relevance
    - Size-constrained extraction methods
    - Subgraph optimization techniques
    
    The extractor provides a unified interface for extracting subgraphs
    that are relevant to a query or context.
    """
    
    def __init__(self, config: Optional[SubgraphExtractorConfig] = None):
        """
        Initialize the subgraph extractor with configuration.
        
        Args:
            config: Configuration for the subgraph extractor. If None, default configuration is used.
                   See config.py for all available configuration options.
        
        Example usage:
            # Using default configuration
            extractor = SubgraphExtractor()
            
            # Using a preset configuration
            extractor = SubgraphExtractor(ExtractorPresets.large_graph())
            
            # Loading from file
            config = load_config_from_file('path/to/config.yaml')
            extractor = SubgraphExtractor(config)
            
            # Custom configuration
            custom_config = SubgraphExtractorConfig(
                max_nodes=150,
                relevance_threshold=0.4,
                optimize_subgraph=True
            )
            extractor = SubgraphExtractor(custom_config)
        """
        # Use default config if none provided
        self.config = config or SubgraphExtractorConfig()
        
        # Extract main parameters for convenience
        self.max_nodes = self.config.max_nodes
        self.max_edges = self.config.max_edges
        self.max_density = self.config.max_density
        self.relevance_threshold = self.config.relevance_threshold
        self.context_aware = self.config.context_aware
        self.optimize_subgraph = self.config.optimize_subgraph
        self.embedding_attr = self.config.embedding_attr
        self.text_attr = self.config.text_attr
        self.preserve_seed_nodes = self.config.preserve_seed_nodes
        
        # Initialize component objects with their configurations
        scorer_config = self.config.scorer_config
        self.scorer = NodeEdgeScorer(
            embedding_attr=scorer_config.embedding_attr,
            text_attr=scorer_config.text_attr,
            importance_attr=scorer_config.importance_attr,
            alpha=scorer_config.alpha,
            beta=scorer_config.beta,
            gamma=scorer_config.gamma,
            min_score=scorer_config.min_score,
            normalize_scores=scorer_config.normalize_scores
        )
        
        pruner_config = self.config.pruner_config
        self.pruner = ContextPruner(
            node_threshold=pruner_config.node_threshold,
            edge_threshold=pruner_config.edge_threshold,
            preserve_connectivity=pruner_config.preserve_connectivity,
            preserve_seed_nodes=pruner_config.preserve_seed_nodes,
            preserve_edge_weight_attr=pruner_config.preserve_edge_weight_attr,
            max_edge_distance=pruner_config.max_edge_distance
        )
        
        constrainer_config = self.config.constrainer_config
        self.constrainer = SizeConstrainer(
            max_nodes=constrainer_config.max_nodes,
            max_edges=constrainer_config.max_edges,
            max_density=constrainer_config.max_density,
            prioritize_by=constrainer_config.prioritize_by,
            preserve_seed_nodes=constrainer_config.preserve_seed_nodes,
            balance_threshold=constrainer_config.balance_threshold
        )
        
        optimizer_config = self.config.optimizer_config
        self.optimizer = SubgraphOptimizer(
            relevance_weight=optimizer_config.relevance_weight,
            diversity_weight=optimizer_config.diversity_weight,
            connectivity_weight=optimizer_config.connectivity_weight,
            max_iterations=optimizer_config.max_iterations,
            improvement_threshold=optimizer_config.improvement_threshold,
            random_seed=optimizer_config.random_seed
        )
        
        logger.info(f"Initialized SubgraphExtractor with max_nodes={self.max_nodes}, "
                   f"relevance_threshold={self.relevance_threshold}")
    
    def extract_subgraph(self, 
                         graph: nx.Graph, 
                         query_embedding: Optional[np.ndarray] = None,
                         query_text: Optional[str] = None,
                         seed_nodes: Optional[List[Any]] = None,
                         extraction_method: str = 'relevance',
                         max_distance: int = 3,
                         node_types: Optional[List[str]] = None,
                         edge_types: Optional[List[str]] = None,
                         use_constraints: bool = True,
                         optimize_result: bool = True) -> Tuple[nx.Graph, Dict[Any, float]]:
        """
        Extract a relevant subgraph from the knowledge graph.
        
        This method combines all extraction components to produce a high-quality
        subgraph that is relevant to the query context.
        
        Args:
            graph: NetworkX graph to extract from
            query_embedding: Query embedding vector for similarity
            query_text: Query text (alternative to embedding)
            seed_nodes: Starting nodes for extraction
            extraction_method: Method for extraction ('relevance', 'distance', 'community')
            max_distance: Maximum distance for distance-based methods
            node_types: Node types to include in extraction
            edge_types: Edge types to include in extraction
            use_constraints: Whether to apply size constraints
            optimize_result: Whether to optimize the final subgraph
            
        Returns:
            Tuple[nx.Graph, Dict[Any, float]]: (Extracted subgraph, node relevance scores)
        """
        # Validate inputs
        if graph.number_of_nodes() == 0:
            logger.warning("Input graph is empty. Cannot extract subgraph.")
            return nx.Graph(), {}
            
        if not query_embedding and not query_text and not seed_nodes:
            logger.warning("No query context provided. Using generic extraction.")
            if graph.number_of_nodes() <= self.max_nodes:
                return graph, {node: 1.0 for node in graph.nodes()}
        
        # Convert query text to embedding if needed
        if not query_embedding and query_text:
            logger.warning("Query text provided but no embedding. Using seed nodes only.")
        
        # Ensure seed_nodes is a list
        if seed_nodes is None:
            seed_nodes = []
        
        # Filter seed nodes to those actually in the graph
        seed_nodes = [node for node in seed_nodes if node in graph]
        
        # Step 1: Score nodes and edges for relevance
        node_scores, edge_scores = self.scorer.score_graph(graph, query_embedding)
        
        # Step 2: Apply context-aware pruning
        if self.context_aware and extraction_method == 'relevance':
            # Prune based on relevance scores
            pruned_graph = self.pruner.prune_by_relevance(
                graph, node_scores, edge_scores, seed_nodes)
            
            # Filter scores to nodes/edges in pruned graph
            pruned_node_scores = {node: score for node, score in node_scores.items() 
                               if node in pruned_graph}
            
            # Update graph for next steps
            working_graph = pruned_graph
        elif extraction_method == 'distance' and seed_nodes:
            # Prune based on distance from seed nodes
            pruned_graph = self.pruner.prune_by_distance(
                graph, seed_nodes, max_distance)
            
            # Filter scores to nodes in pruned graph
            pruned_node_scores = {node: score for node, score in node_scores.items() 
                               if node in pruned_graph}
            
            # Update graph for next steps
            working_graph = pruned_graph
        elif extraction_method == 'community' and seed_nodes:
            # Extract community containing seed nodes
            from networkx.algorithms import community
            
            # Start with neighborhood of seed nodes
            neighborhood = set(seed_nodes)
            for node in seed_nodes:
                neighborhood.update(graph.neighbors(node))
            
            # Extract subgraph
            community_graph = graph.subgraph(neighborhood).copy()
            
            # Filter scores to nodes in community
            pruned_node_scores = {node: score for node, score in node_scores.items() 
                               if node in community_graph}
            
            # Update graph for next steps
            working_graph = community_graph
        else:
            # No pruning, use full graph
            working_graph = graph
            pruned_node_scores = node_scores
        
        # Step 3: Apply type-based filtering if requested
        if node_types or edge_types:
            filtered_graph = working_graph.copy()
            
            # Filter nodes by type
            if node_types:
                nodes_to_remove = []
                for node in filtered_graph.nodes():
                    node_type = filtered_graph.nodes[node].get('type')
                    if node_type not in node_types and node not in seed_nodes:
                        nodes_to_remove.append(node)
                
                filtered_graph.remove_nodes_from(nodes_to_remove)
            
            # Filter edges by type
            if edge_types:
                edges_to_remove = []
                for u, v, data in filtered_graph.edges(data=True):
                    edge_type = data.get('type')
                    if edge_type not in edge_types:
                        edges_to_remove.append((u, v))
                
                filtered_graph.remove_edges_from(edges_to_remove)
            
            # Update working graph and scores
            working_graph = filtered_graph
            pruned_node_scores = {node: score for node, score in pruned_node_scores.items() 
                              if node in working_graph}
        
        # Step 4: Apply size constraints
        if use_constraints and (working_graph.number_of_nodes() > self.max_nodes or 
                              working_graph.number_of_edges() > self.max_edges):
            # Apply size and density constraints
            constrained_graph = self.constrainer.constrain_by_size(
                working_graph, seed_nodes, pruned_node_scores)
            
            # Ensure connectivity if needed
            if self.preserve_seed_nodes and seed_nodes:
                constrained_graph = self.constrainer.ensure_connectedness(
                    constrained_graph, seed_nodes)
            
            # Apply density constraint if needed
            if (constrained_graph.number_of_nodes() > 1 and 
                constrained_graph.number_of_edges() / 
                (constrained_graph.number_of_nodes() * (constrained_graph.number_of_nodes() - 1) / 2) 
                > self.max_density):
                constrained_graph = self.constrainer.constrain_by_density(
                    constrained_graph, pruned_node_scores, seed_nodes)
            
            # Update working graph and scores
            working_graph = constrained_graph
            pruned_node_scores = {node: score for node, score in pruned_node_scores.items() 
                              if node in working_graph}
        
        # Step 5: Optimize subgraph if requested
        if optimize_result and self.optimize_subgraph:
            # Get edge scores for current graph
            current_edge_scores = {(u, v): edge_scores.get((u, v), 0.0)
                                for u, v in working_graph.edges()}
            
            # Apply hybrid optimization
            optimized_graph = self.optimizer.optimize_hybrid(
                working_graph, pruned_node_scores, current_edge_scores, seed_nodes)
            
            # Update working graph and scores
            working_graph = optimized_graph
            pruned_node_scores = {node: score for node, score in pruned_node_scores.items() 
                              if node in working_graph}
        
        # Final size check
        if working_graph.number_of_nodes() > self.max_nodes:
            logger.warning(f"Final graph still exceeds max_nodes ({working_graph.number_of_nodes()} > {self.max_nodes}). "
                          "Applying final size constraint.")
            working_graph = self.constrainer.constrain_by_size(
                working_graph, seed_nodes, pruned_node_scores)
            
            pruned_node_scores = {node: score for node, score in pruned_node_scores.items() 
                              if node in working_graph}
        
        # Log extraction results
        logger.info(f"Extracted subgraph with {working_graph.number_of_nodes()} nodes and "
                   f"{working_graph.number_of_edges()} edges using method '{extraction_method}'")
        
        return working_graph, pruned_node_scores
    
    def extract_multi_query_subgraph(self, 
                                   graph: nx.Graph, 
                                   query_embeddings: List[np.ndarray],
                                   seed_nodes_list: Optional[List[List[Any]]] = None,
                                   merge_method: str = 'union',
                                   balance_factor: float = 0.5) -> nx.Graph:
        """
        Extract a subgraph relevant to multiple queries.
        
        Args:
            graph: NetworkX graph to extract from
            query_embeddings: List of query embedding vectors
            seed_nodes_list: List of seed node lists for each query
            merge_method: Method for merging subgraphs ('union', 'intersection', 'weighted')
            balance_factor: Weight between union and intersection for 'weighted' method
            
        Returns:
            nx.Graph: Merged subgraph
        """
        if not query_embeddings:
            logger.warning("No query embeddings provided. Cannot extract multi-query subgraph.")
            return nx.Graph()
        
        # Ensure seed_nodes_list matches query_embeddings
        if seed_nodes_list is None:
            seed_nodes_list = [[] for _ in query_embeddings]
        elif len(seed_nodes_list) != len(query_embeddings):
            logger.warning("seed_nodes_list length doesn't match query_embeddings length. Using empty seed lists.")
            seed_nodes_list = [[] for _ in query_embeddings]
        
        # Extract subgraph for each query
        subgraphs = []
        scores_list = []
        
        for i, embedding in enumerate(query_embeddings):
            seed_nodes = seed_nodes_list[i]
            subgraph, scores = self.extract_subgraph(
                graph, embedding, seed_nodes=seed_nodes)
            
            subgraphs.append(subgraph)
            scores_list.append(scores)
        
        # Merge subgraphs according to merge method
        if merge_method == 'union':
            # Take union of all subgraphs
            merged_graph: nx.Graph = nx.Graph()
            
            for subgraph in subgraphs:
                for node in subgraph.nodes:
                    if not merged_graph.has_node(node):
                        merged_graph.add_node(node, **graph.nodes[node])
                
                for u, v, data in subgraph.edges(data=True):
                    if not merged_graph.has_edge(u, v):
                        merged_graph.add_edge(u, v, **data)
        
        elif merge_method == 'intersection':
            # Take intersection of all subgraphs
            if not subgraphs:
                return nx.Graph()
                
            # Start with nodes in first subgraph
            common_nodes = set(subgraphs[0].nodes())
            
            # Intersect with remaining subgraphs
            for subgraph in subgraphs[1:]:
                common_nodes.intersection_update(subgraph.nodes())
            
            # Create intersection graph
            merged_graph = graph.subgraph(common_nodes).copy()
        
        elif merge_method == 'weighted':
            # Weighted combination of union and intersection
            # Start with all nodes from all subgraphs
            all_nodes = set()
            node_counts: Dict[Any, int] = {}  # How many subgraphs contain this node
            
            for subgraph in subgraphs:
                for node in subgraph.nodes:
                    all_nodes.add(node)
                    node_counts[node] = node_counts.get(node, 0) + 1
            
            # Calculate weighted scores based on occurrence count
            # Higher scores for nodes appearing in more subgraphs
            weighted_scores = {}
            for node in all_nodes:
                count_weight = node_counts[node] / len(subgraphs)  # Normalized count
                
                # Average relevance scores across subgraphs containing this node
                avg_score = 0.0
                score_count = 0
                
                for scores in scores_list:
                    if node in scores:
                        avg_score += scores[node]
                        score_count += 1
                
                if score_count > 0:
                    avg_score /= score_count
                
                # Weighted score combines occurrence and relevance
                weighted_scores[node] = (balance_factor * count_weight + 
                                        (1 - balance_factor) * avg_score)
            
            # Sort nodes by weighted score
            sorted_nodes = sorted(
                weighted_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Take top nodes within size constraint
            top_nodes = [node for node, _ in sorted_nodes[:self.max_nodes]]
            
            # Create merged graph
            merged_graph = graph.subgraph(top_nodes).copy()
        
        else:
            logger.warning(f"Unknown merge method: {merge_method}. Using union.")
            return self.extract_multi_query_subgraph(
                graph, query_embeddings, seed_nodes_list, 'union', balance_factor)
        
        # Apply any final constraints
        if merged_graph.number_of_nodes() > self.max_nodes:
            # Calculate aggregate scores for final constraint
            aggregate_scores = {}
            
            for node in merged_graph.nodes:
                # Average across all score sets
                scores_sum = 0.0
                count = 0
                
                for scores in scores_list:
                    if node in scores:
                        scores_sum += scores[node]
                        count += 1
                
                if count > 0:
                    aggregate_scores[node] = scores_sum / count
                else:
                    aggregate_scores[node] = 0.0
            
            # Collect all seed nodes
            all_seeds = set()
            for seeds in seed_nodes_list:
                all_seeds.update(seeds)
            
            # Apply size constraint
            merged_graph = self.constrainer.constrain_by_size(
                merged_graph, list(all_seeds), aggregate_scores)
        
        logger.info(f"Merged multi-query subgraph contains {merged_graph.number_of_nodes()} nodes "
                   f"and {merged_graph.number_of_edges()} edges using method '{merge_method}'")
        
        return merged_graph
    
    def extract_neighborhood(self, 
                           graph: nx.Graph, 
                           center_nodes: List[Any],
                           radius: int = 2,
                           limit_per_node: int = 10,
                           weighted: bool = False,
                           weight_attr: str = 'weight') -> nx.Graph:
        """
        Extract a neighborhood subgraph around center nodes.
        
        Args:
            graph: NetworkX graph to extract from
            center_nodes: Center nodes for the neighborhood
            radius: Maximum distance from center nodes
            limit_per_node: Maximum neighbors to consider per node
            weighted: Whether to use edge weights for traversal
            weight_attr: Edge attribute to use as weight
            
        Returns:
            nx.Graph: Neighborhood subgraph
        """
        # Create a new graph for the neighborhood
        neighborhood: nx.Graph = nx.Graph()
        
        # Add center nodes to neighborhood
        for node in center_nodes:
            if node in graph:
                neighborhood.add_node(node, **graph.nodes[node])
        
        # For each center node, add its neighborhood
        for center in center_nodes:
            if center not in graph:
                continue
            
            # Collect nodes at each distance
            visited = {center: 0}  # node -> distance
            level_nodes = {0: [center]}
            
            # BFS traversal up to radius
            for level in range(1, radius + 1):
                level_nodes[level] = []
                
                # Process previous level's nodes
                for node in level_nodes[level - 1]:
                    # Get neighbors sorted by weight if weighted
                    neighbors = list(graph.neighbors(node))
                    
                    if weighted and weight_attr:
                        # Get neighbors with weights
                        weighted_neighbors = []
                        for neighbor in neighbors:
                            if graph.has_edge(node, neighbor):
                                weight = graph.edges[node, neighbor].get(weight_attr, 1.0)
                                weighted_neighbors.append((neighbor, weight))
                        
                        # Sort by weight (descending)
                        weighted_neighbors.sort(key=lambda x: x[1], reverse=True)
                        neighbors = [n for n, _ in weighted_neighbors]
                    
                    # Limit number of neighbors per node
                    neighbors = neighbors[:limit_per_node]
                    
                    # Process neighbors
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            # Mark as visited
                            visited[neighbor] = level
                            level_nodes[level].append(neighbor)
                            
                            # Add to neighborhood graph
                            neighborhood.add_node(neighbor, **graph.nodes[neighbor])
                            
                            # Add edge to parent
                            if graph.has_edge(node, neighbor):
                                neighborhood.add_edge(node, neighbor, **graph.edges[node, neighbor])
            
            # Connect center nodes directly if they're not already connected
            for i, c1 in enumerate(center_nodes):
                for c2 in center_nodes[i+1:]:
                    if (c1 in neighborhood and c2 in neighborhood and 
                        not neighborhood.has_edge(c1, c2) and graph.has_edge(c1, c2)):
                        neighborhood.add_edge(c1, c2, **graph.edges[c1, c2])
        
        logger.info(f"Extracted neighborhood with {neighborhood.number_of_nodes()} nodes and "
                   f"{neighborhood.number_of_edges()} edges around {len(center_nodes)} center nodes")
        
        return neighborhood
