"""
Graph Search Engine

Provides a unified interface for graph search operations in GraphRAG.
"""

import logging
import time
from typing import Dict, List, Set, Any, Optional, Callable, Tuple, Union
import networkx as nx
import numpy as np

from .traversal import GraphTraversal
from .path_finder import PathFinder
from .relevance_search import RelevanceSearch
from .optimized_search import OptimizedGraphSearch

logger = logging.getLogger(__name__)

class GraphSearchEngine:
    """
    Main search engine for knowledge graphs in GraphRAG.
    
    This class provides a unified interface for various graph search operations,
    including traversal, path finding, relevance search, and optimized search.
    """
    
    def __init__(self, 
                 max_depth: int = 5,
                 max_nodes: int = 1000, 
                 max_paths: int = 10,
                 relevance_threshold: float = 0.5,
                 use_optimization: bool = True,
                 parallel_workers: int = 4,
                 enable_caching: bool = True):
        """
        Initialize the graph search engine with configurable parameters.
        
        Args:
            max_depth: Maximum traversal/search depth
            max_nodes: Maximum nodes to consider in traversal
            max_paths: Maximum paths to return in path finding
            relevance_threshold: Minimum relevance score for results
            use_optimization: Whether to use optimized search algorithms
            parallel_workers: Number of parallel workers for optimized search
            enable_caching: Whether to enable result caching
        """
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.max_paths = max_paths
        self.relevance_threshold = relevance_threshold
        self.use_optimization = use_optimization
        self.parallel_workers = parallel_workers
        self.enable_caching = enable_caching
        
        # Initialize component modules
        self.traversal = GraphTraversal(max_depth=max_depth, max_nodes=max_nodes)
        self.path_finder = PathFinder(max_paths=max_paths, max_path_length=max_depth)
        self.relevance_search = RelevanceSearch(
            max_results=max_nodes, relevance_threshold=relevance_threshold)
        
        # Optimized search component that may be None if optimization is disabled
        self.optimized: Optional[OptimizedGraphSearch] = None
        
        if use_optimization:
            self.optimized = OptimizedGraphSearch(
                max_workers=parallel_workers,
                use_parallel=parallel_workers > 1,
                cache_results=enable_caching)
        else:
            self.optimized = None
        
        logger.info(f"Initialized GraphSearchEngine with max_depth={max_depth}, "
                   f"max_nodes={max_nodes}, optimization={use_optimization}")
    
    def bfs(self, 
           graph: nx.Graph, 
           start_node: Any, 
           **kwargs: Any) -> Dict[Any, Tuple[int, List[Any]]]:
        """
        Perform breadth-first search traversal.
        
        Args:
            graph: NetworkX graph to traverse
            start_node: Starting node for traversal
            **kwargs: Additional parameters for traversal
            
        Returns:
            Dict[Any, Tuple[int, List[Any]]]: Map of node -> (depth, path)
        """
        if self.use_optimization and self.optimized is not None:
            # Use optimized implementation
            max_depth = kwargs.get('max_depth', self.max_depth)
            max_nodes = kwargs.get('max_nodes', self.max_nodes)
            return self.optimized.optimized_bfs(
                graph, start_node, max_depth=max_depth, max_nodes=max_nodes)
        
        # Use standard implementation
        results = {}
        for node, depth, path in self.traversal.breadth_first_search(
            graph, start_node, **kwargs):
            results[node] = (depth, path)
        
        return results
    
    def dfs(self, 
           graph: nx.Graph, 
           start_node: Any, 
           **kwargs: Any) -> Dict[Any, Tuple[int, List[Any]]]:
        """
        Perform depth-first search traversal.
        
        Args:
            graph: NetworkX graph to traverse
            start_node: Starting node for traversal
            **kwargs: Additional parameters for traversal
            
        Returns:
            Dict[Any, Tuple[int, List[Any]]]: Map of node -> (depth, path)
        """
        results = {}
        for node, depth, path in self.traversal.depth_first_search(
            graph, start_node, **kwargs):
            results[node] = (depth, path)
        
        return results
    
    def find_path(self, 
                 graph: nx.Graph, 
                 source: Any, 
                 target: Any, 
                 method: str = 'shortest', 
                 **kwargs: Any) -> Union[List[Any], List[Tuple[List[Any], float]]]:
        """
        Find a path or paths between two nodes.
        
        Args:
            graph: NetworkX graph to search
            source: Source node
            target: Target node
            method: Path finding method ('shortest', 'all_shortest', 'k_shortest',
                   'weighted', 'semantic', 'diverse')
            **kwargs: Additional parameters for path finding
            
        Returns:
            Union[List[Any], List[Tuple[List[Any], float]]]: Path(s) between nodes
        """
        # For optimized shortest paths to multiple targets
        if method == 'shortest' and self.use_optimization and self.optimized is not None and 'targets' in kwargs:
            targets = kwargs.pop('targets')
            weight = kwargs.get('weight')
            cutoff = kwargs.get('cutoff')
            # Get the paths dictionary from optimized search
            path_dict = self.optimized.optimized_shortest_paths(
                graph, source, targets, weight=weight, cutoff=cutoff)
            # Convert to list format to match return type
            return [path for path in path_dict.values() if path]
        
        # Regular path finding
        if method == 'shortest':
            return self.path_finder.shortest_path(graph, source, target, **kwargs) or []
        elif method == 'all_shortest':
            return self.path_finder.all_shortest_paths(graph, source, target, **kwargs)
        elif method == 'k_shortest':
            return self.path_finder.k_shortest_paths(graph, source, target, **kwargs)
        elif method == 'weighted':
            return self.path_finder.weighted_paths(graph, source, target, **kwargs)
        elif method == 'semantic':
            return self.path_finder.semantic_paths(graph, source, target, **kwargs)
        elif method == 'diverse':
            return self.path_finder.diverse_paths(graph, source, target, **kwargs)
        elif method == 'bidirectional':
            return self.traversal.bidirectional_search(graph, source, target, **kwargs) or []
        else:
            raise ValueError(f"Unknown path finding method: {method}")
    
    def search_by_relevance(self, 
                          graph: nx.Graph, 
                          query_vector: np.ndarray, 
                          method: str = 'embedding', 
                          **kwargs: Any) -> List[Tuple[Any, float]]:
        """
        Search for nodes based on relevance to a query.
        
        Args:
            graph: NetworkX graph to search
            query_vector: Query embedding vector
            method: Relevance search method ('embedding', 'pagerank', 'activation', 'expansion')
            **kwargs: Additional parameters for relevance search
            
        Returns:
            List[Tuple[Any, float]]: List of (node, relevance) tuples
        """
        if method == 'embedding' and self.use_optimization and self.optimized is not None:
            # Use optimized implementation
            embedding_attr = kwargs.get('embedding_attr', 'embedding')
            min_similarity = kwargs.get('min_similarity', self.relevance_threshold)
            max_results = kwargs.get('max_results', self.max_nodes)
            return self.optimized.optimized_embedding_search(
                graph, query_vector, embedding_attr=embedding_attr,
                min_similarity=min_similarity, max_results=max_results)
        
        if method == 'embedding':
            return self.relevance_search.search_by_embedding(
                graph, query_vector, **kwargs)
        elif method == 'pagerank':
            seed_nodes = kwargs.pop('seed_nodes')
            return list(self.relevance_search.personalized_pagerank(
                graph, seed_nodes, **kwargs).items())
        elif method == 'activation':
            seed_nodes = kwargs.pop('seed_nodes')
            return list(self.relevance_search.spread_activation_search(
                graph, seed_nodes, **kwargs).items())
        elif method == 'expansion':
            seed_nodes = kwargs.pop('seed_nodes')
            nodes = self.relevance_search.semantic_expansion(
                graph, seed_nodes, query_vector=query_vector, **kwargs)
            # Convert to (node, score) format for consistency
            return [(node, 1.0) for node in nodes]
        else:
            raise ValueError(f"Unknown relevance search method: {method}")
    
    def extract_subgraph(self, 
                        graph: nx.Graph, 
                        center_nodes: List[Any], 
                        method: str = 'neighborhood', 
                        **kwargs: Any) -> nx.Graph:
        """
        Extract a relevant subgraph from the knowledge graph.
        
        Args:
            graph: NetworkX graph to extract from
            center_nodes: Center nodes for the subgraph
            method: Subgraph extraction method ('neighborhood', 'paths', 'community')
            **kwargs: Additional parameters for subgraph extraction
            
        Returns:
            nx.Graph: Extracted subgraph
        """
        if method == 'neighborhood' and self.use_optimization and self.optimized is not None:
            # Use optimized implementation
            radius = kwargs.get('radius', 2)
            max_nodes = kwargs.get('max_nodes', self.max_nodes)
            return self.optimized.optimized_subgraph_extraction(
                graph, center_nodes, radius=radius, max_nodes=max_nodes)
        
        if method == 'neighborhood':
            # Extract neighborhood within radius
            radius = kwargs.get('radius', 2)
            max_nodes = kwargs.get('max_nodes', self.max_nodes)
            
            # Get all nodes within radius of center nodes
            included_nodes = set(center_nodes)
            
            for center in center_nodes:
                # Use BFS to get neighborhood
                for node, depth, _ in self.traversal.breadth_first_search(
                    graph, center, max_depth=radius, max_nodes=max_nodes):
                    included_nodes.add(node)
                    
                    # Check if we've reached max_nodes
                    if len(included_nodes) >= max_nodes:
                        break
            
            # Extract the subgraph
            return graph.subgraph(included_nodes).copy()
        
        elif method == 'paths':
            # Extract subgraph based on paths between nodes
            max_paths = kwargs.get('max_paths', self.max_paths)
            included_nodes = set(center_nodes)
            
            # Find paths between all pairs of center nodes
            for i, source in enumerate(center_nodes):
                for target in center_nodes[i+1:]:
                    paths = self.find_path(
                        graph, source, target, method='diverse', 
                        max_paths=max_paths, **kwargs)
                    
                    # Add all nodes from paths
                    for path in paths:
                        included_nodes.update(path)
            
            # Extract the subgraph
            return graph.subgraph(included_nodes).copy()
        
        elif method == 'community':
            # Extract community containing center nodes
            from networkx.algorithms import community
            
            # Get largest connected component containing center nodes
            components = list(nx.connected_components(graph.to_undirected()))
            relevant_components = []
            
            for component in components:
                if any(node in component for node in center_nodes):
                    relevant_components.append(component)
            
            if not relevant_components:
                # Fallback to neighborhood method
                return self.extract_subgraph(
                    graph, center_nodes, method='neighborhood', **kwargs)
            
            # Use the largest relevant component
            component = max(relevant_components, key=len)
            
            # Extract the subgraph
            return graph.subgraph(component).copy()
        
        else:
            raise ValueError(f"Unknown subgraph extraction method: {method}")
    
    def find_connecting_nodes(self,
                            graph: nx.Graph,
                            node_groups: List[List[Any]],
                            max_connecting: int = 100) -> List[Any]:
        """
        Find nodes that connect multiple groups of nodes in the graph.
        
        Args:
            graph: NetworkX graph to search
            node_groups: List of node groups to connect
            max_connecting: Maximum number of connecting nodes to return
            
        Returns:
            List[Any]: List of connecting nodes
        """
        if not node_groups or len(node_groups) < 2:
            logger.warning("Need at least two node groups to find connecting nodes")
            return []
        
        # Get all unique nodes from groups
        all_group_nodes = set()
        for group in node_groups:
            all_group_nodes.update(group)
        
        # Initialize node scores based on shortest paths
        node_scores: Dict[Any, float] = {}
        
        # For each pair of groups
        for i, group1 in enumerate(node_groups):
            for group2 in node_groups[i+1:]:
                # Find paths between groups
                for source in group1:
                    for target in group2:
                        path = self.find_path(graph, source, target, method='shortest')
                        
                        if not path:
                            continue
                        
                        # Score internal nodes on the path
                        for node in path[1:-1]:  # Exclude source and target
                            if node not in all_group_nodes:
                                node_scores[node] = node_scores.get(node, 0) + 1
        
        # Sort nodes by score
        connecting_nodes = sorted(
            node_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top nodes
        return [node for node, _ in connecting_nodes[:max_connecting]]
    
    def search_by_attribute(self,
                          graph: nx.Graph,
                          attribute: str,
                          value: Any,
                          **kwargs: Any) -> List[Any]:
        """
        Search for nodes with a specific attribute value.
        
        Args:
            graph: NetworkX graph to search
            attribute: Node attribute to match
            value: Attribute value to match
            **kwargs: Additional parameters for attribute search
            
        Returns:
            List[Any]: List of matching nodes
        """
        if self.use_optimization and self.optimized is not None and hasattr(self.optimized, 'attribute_indices'):
            # Use index-based search if available
            return self.optimized.attribute_index_search(graph, attribute, value)
        
        # Use standard implementation
        return self.relevance_search.search_by_attribute(
            graph, attribute, value, **kwargs)
    
    def prepare_graph(self, graph: nx.Graph) -> None:
        """
        Prepare a graph for efficient search operations.
        
        This performs optimizations like building indices and caches.
        
        Args:
            graph: NetworkX graph to prepare
        """
        if self.use_optimization and self.optimized is not None:
            if hasattr(self.optimized, 'build_indices'):
                # Build attribute indices
                self.optimized.index_attributes = ['type', 'category', 'entity_type']
                self.optimized.build_indices(graph)
                
                logger.info("Prepared graph for efficient search operations")
    
    def clear_caches(self) -> None:
        """Clear all search caches."""
        if self.use_optimization and self.optimized:
            if hasattr(self.optimized, 'clear_cache'):
                self.optimized.clear_cache()
                logger.info("Cleared search caches")
