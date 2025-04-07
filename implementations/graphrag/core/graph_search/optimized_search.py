"""
Optimized Graph Search

Implements optimized search algorithms for efficient operations on large graphs.
"""

import logging
import time
from typing import Dict, List, Set, Any, Optional, Callable, Tuple, Iterator, Union
import networkx as nx
import numpy as np
from collections import defaultdict
import heapq
import threading
import concurrent.futures

logger = logging.getLogger(__name__)

class OptimizedGraphSearch:
    """
    Provides performance-optimized search capabilities for large knowledge graphs.
    
    This class implements optimized versions of graph search algorithms,
    including parallel processing, indexing, and search space pruning techniques.
    """
    
    def __init__(self, 
                 max_workers: int = 4,
                 use_parallel: bool = True,
                 cache_results: bool = True,
                 cache_ttl: int = 3600,
                 index_attributes: Optional[List[str]] = None):
        """
        Initialize the optimized graph search engine.
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
            use_parallel: Whether to use parallel processing for search operations
            cache_results: Whether to cache search results
            cache_ttl: Time-to-live for cached results in seconds
            index_attributes: List of node attributes to index for faster lookup
        """
        self.max_workers = max_workers
        self.use_parallel = use_parallel
        self.cache_results = cache_results
        self.cache_ttl = cache_ttl
        self.index_attributes = index_attributes or []
        
        # Initialize result cache
        self.result_cache = {}
        self.cache_timestamps = {}
        self.cache_lock = threading.RLock()
        
        # Initialize attribute indices
        self.attribute_indices = {}
        
        logger.info(f"Initialized OptimizedGraphSearch with max_workers={max_workers}, "
                   f"use_parallel={use_parallel}, cache_results={cache_results}")
    
    def build_indices(self, graph: nx.Graph) -> None:
        """
        Build indices for specified node attributes to optimize search operations.
        
        Args:
            graph: NetworkX graph to index
        """
        start_time = time.time()
        
        # Build indices for each specified attribute
        for attr in self.index_attributes:
            # Create index: attribute value -> list of nodes
            index = defaultdict(list)
            
            # Populate index
            for node, data in graph.nodes(data=True):
                if attr in data:
                    value = data[attr]
                    
                    # Handle both single values and collections
                    if isinstance(value, (list, tuple, set)):
                        for val in value:
                            index[val].append(node)
                    else:
                        index[value].append(node)
            
            # Store index
            self.attribute_indices[attr] = dict(index)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Built indices for {len(self.index_attributes)} attributes in {elapsed_time:.3f}s")
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        with self.cache_lock:
            self.result_cache = {}
            self.cache_timestamps = {}
        logger.info("Cleared search result cache")
    
    def _clean_expired_cache(self) -> None:
        """Remove expired entries from the cache."""
        with self.cache_lock:
            current_time = time.time()
            expired_keys = []
            
            for key, timestamp in self.cache_timestamps.items():
                if current_time - timestamp > self.cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                if key in self.result_cache:
                    del self.result_cache[key]
                del self.cache_timestamps[key]
            
            if expired_keys:
                logger.debug(f"Removed {len(expired_keys)} expired entries from cache")
    
    def _cache_key(self, operation: str, **kwargs) -> str:
        """
        Generate a cache key for an operation with parameters.
        
        Args:
            operation: Name of the search operation
            **kwargs: Operation parameters
            
        Returns:
            str: Cache key string
        """
        # Create string representation of parameters
        params = []
        for key, value in sorted(kwargs.items()):
            # Handle different value types
            if isinstance(value, (list, tuple, set)):
                param_str = f"{key}=[{','.join(str(v) for v in value)}]"
            elif isinstance(value, dict):
                param_str = f"{key}={{{','.join(f'{k}:{v}' for k, v in value.items())}}}"
            else:
                param_str = f"{key}={value}"
            
            params.append(param_str)
        
        return f"{operation}:{':'.join(params)}"
    
    def _get_cached_result(self, operation: str, **kwargs) -> Optional[Any]:
        """
        Get a result from the cache if available and not expired.
        
        Args:
            operation: Name of the search operation
            **kwargs: Operation parameters
            
        Returns:
            Optional[Any]: Cached result or None if not found
        """
        if not self.cache_results:
            return None
        
        with self.cache_lock:
            # Clean expired cache entries
            self._clean_expired_cache()
            
            # Generate cache key
            key = self._cache_key(operation, **kwargs)
            
            # Return cached result if available
            if key in self.result_cache:
                logger.debug(f"Cache hit for {operation}")
                return self.result_cache[key]
            
            logger.debug(f"Cache miss for {operation}")
            return None
    
    def _cache_result(self, operation: str, result: Any, **kwargs) -> None:
        """
        Store a result in the cache.
        
        Args:
            operation: Name of the search operation
            result: Result to cache
            **kwargs: Operation parameters
        """
        if not self.cache_results:
            return
        
        with self.cache_lock:
            # Generate cache key
            key = self._cache_key(operation, **kwargs)
            
            # Store result and timestamp
            self.result_cache[key] = result
            self.cache_timestamps[key] = time.time()
            
            logger.debug(f"Cached result for {operation}")
    
    def _partition_graph(self, graph: nx.Graph, num_partitions: int) -> List[Set[Any]]:
        """
        Partition a graph into roughly equal parts for parallel processing.
        
        Args:
            graph: NetworkX graph to partition
            num_partitions: Number of partitions to create
            
        Returns:
            List[Set[Any]]: List of node sets, one for each partition
        """
        nodes = list(graph.nodes())
        partition_size = max(1, len(nodes) // num_partitions)
        
        partitions = []
        for i in range(0, len(nodes), partition_size):
            partition = set(nodes[i:i + partition_size])
            partitions.append(partition)
        
        return partitions
    
    def optimized_bfs(self,
                     graph: nx.Graph,
                     start_node: Any,
                     max_depth: int = 3,
                     max_nodes: int = 1000) -> Dict[Any, Tuple[int, List[Any]]]:
        """
        Optimized breadth-first search implementation for large graphs.
        
        Args:
            graph: NetworkX graph to search
            start_node: Starting node for the search
            max_depth: Maximum search depth
            max_nodes: Maximum nodes to explore
            
        Returns:
            Dict[Any, Tuple[int, List[Any]]]: Map of node -> (depth, path)
        """
        # Check cache first
        cached_result = self._get_cached_result(
            "optimized_bfs", graph_id=id(graph), start_node=start_node,
            max_depth=max_depth, max_nodes=max_nodes)
        
        if cached_result is not None:
            return cached_result
        
        # Initialize search
        results = {start_node: (0, [start_node])}
        visited = {start_node}
        current_level = {start_node}
        current_depth = 0
        
        # BFS levels
        while current_level and current_depth < max_depth and len(visited) < max_nodes:
            next_level = set()
            
            # Process all nodes at current depth
            for node in current_level:
                current_path = results[node][1]
                
                # Add neighbors to next level
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
                        results[neighbor] = (current_depth + 1, current_path + [neighbor])
                        
                        # Check if we've reached max_nodes
                        if len(visited) >= max_nodes:
                            break
            
            # Move to next depth level
            current_level = next_level
            current_depth += 1
        
        # Cache and return results
        self._cache_result(
            "optimized_bfs", results, graph_id=id(graph), start_node=start_node,
            max_depth=max_depth, max_nodes=max_nodes)
        
        return results
    
    def parallel_subgraph_search(self,
                               graph: nx.Graph,
                               query_function: Callable[[nx.Graph, Set[Any]], Dict[Any, float]],
                               max_results: int = 100) -> Dict[Any, float]:
        """
        Execute a search operation in parallel across graph partitions.
        
        Args:
            graph: NetworkX graph to search
            query_function: Function that takes (graph, node_set) and returns dict of node->score
            max_results: Maximum results to return
            
        Returns:
            Dict[Any, float]: Combined results from all partitions
        """
        if not self.use_parallel or self.max_workers <= 1:
            # Run non-parallel version
            return query_function(graph, set(graph.nodes()))
        
        # Partition the graph
        partitions = self._partition_graph(graph, self.max_workers)
        
        # Function to process a partition
        def process_partition(partition):
            return query_function(graph, partition)
        
        # Execute in parallel
        all_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_partition = {
                executor.submit(process_partition, partition): i 
                for i, partition in enumerate(partitions)
            }
            
            for future in concurrent.futures.as_completed(future_to_partition):
                partition_index = future_to_partition[future]
                try:
                    partition_results = future.result()
                    all_results.update(partition_results)
                except Exception as e:
                    logger.error(f"Error processing partition {partition_index}: {e}")
        
        # Sort and limit results
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        limited_results = sorted_results[:max_results]
        
        return dict(limited_results)
    
    def optimized_shortest_paths(self,
                               graph: nx.Graph,
                               source: Any,
                               targets: List[Any],
                               weight: Optional[str] = None,
                               cutoff: Optional[int] = None) -> Dict[Any, List[Any]]:
        """
        Find shortest paths from source to multiple targets efficiently.
        
        Args:
            graph: NetworkX graph to search
            source: Source node
            targets: List of target nodes
            weight: Edge attribute to use as weight
            cutoff: Maximum path length
            
        Returns:
            Dict[Any, List[Any]]: Map of target -> path
        """
        # Check cache first
        cache_key = (source, tuple(sorted(targets)), weight, cutoff)
        cached_result = self._get_cached_result(
            "optimized_shortest_paths", graph_id=id(graph), cache_key=cache_key)
        
        if cached_result is not None:
            return cached_result
        
        # For small number of targets, use separate path calculations
        if len(targets) <= 5:
            results = {}
            for target in targets:
                try:
                    path = nx.shortest_path(graph, source=source, target=target, weight=weight)
                    if cutoff is None or len(path) - 1 <= cutoff:
                        results[target] = path
                except nx.NetworkXNoPath:
                    pass
            
            # Cache and return results
            self._cache_result(
                "optimized_shortest_paths", results, graph_id=id(graph), cache_key=cache_key)
            
            return results
        
        # For larger number of targets, do a single-source shortest paths calculation
        try:
            paths = nx.single_source_shortest_path(graph, source, cutoff=cutoff)
            results = {target: paths[target] for target in targets if target in paths}
            
            # Cache and return results
            self._cache_result(
                "optimized_shortest_paths", results, graph_id=id(graph), cache_key=cache_key)
            
            return results
        except nx.NetworkXError as e:
            logger.error(f"Error finding shortest paths: {e}")
            return {}
    
    def optimized_embedding_search(self,
                                 graph: nx.Graph,
                                 query_vector: np.ndarray,
                                 embedding_attr: str = 'embedding',
                                 min_similarity: float = 0.5,
                                 max_results: int = 100) -> List[Tuple[Any, float]]:
        """
        Optimized search for nodes based on embedding similarity.
        
        Args:
            graph: NetworkX graph to search
            query_vector: Query embedding vector
            embedding_attr: Node attribute containing embedding vectors
            min_similarity: Minimum similarity threshold
            max_results: Maximum results to return
            
        Returns:
            List[Tuple[Any, float]]: List of (node, similarity) tuples
        """
        # Check cache first
        query_hash = hash(tuple(query_vector.flatten().tolist()))
        cached_result = self._get_cached_result(
            "optimized_embedding_search", graph_id=id(graph), query_hash=query_hash,
            embedding_attr=embedding_attr, min_similarity=min_similarity, max_results=max_results)
        
        if cached_result is not None:
            return cached_result
        
        # Parallel implementation for large graphs
        if self.use_parallel and len(graph) > 1000:
            def process_nodes(graph, nodes):
                results = []
                for node in nodes:
                    node_data = graph.nodes[node]
                    if embedding_attr not in node_data:
                        continue
                    
                    embedding = node_data[embedding_attr]
                    if not isinstance(embedding, np.ndarray):
                        embedding = np.array(embedding)
                    
                    # Normalize vectors for cosine similarity
                    query_norm = query_vector / np.linalg.norm(query_vector)
                    embedding_norm = embedding / np.linalg.norm(embedding)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_norm, embedding_norm)
                    
                    # Add to results if above threshold
                    if similarity >= min_similarity:
                        results.append((node, float(similarity)))
                
                return results
            
            # Split nodes into batches
            all_nodes = list(graph.nodes())
            batch_size = len(all_nodes) // self.max_workers
            batches = [all_nodes[i:i+batch_size] for i in range(0, len(all_nodes), batch_size)]
            
            # Process in parallel
            all_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_batch = {
                    executor.submit(process_nodes, graph, batch): i 
                    for i, batch in enumerate(batches)
                }
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)
                    except Exception as e:
                        logger.error(f"Error processing node batch: {e}")
            
            # Sort and limit results
            all_results.sort(key=lambda x: x[1], reverse=True)
            results = all_results[:max_results]
        else:
            # Sequential implementation for smaller graphs
            results = []
            for node in graph.nodes():
                node_data = graph.nodes[node]
                if embedding_attr not in node_data:
                    continue
                
                embedding = node_data[embedding_attr]
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                
                # Normalize vectors for cosine similarity
                query_norm = query_vector / np.linalg.norm(query_vector)
                embedding_norm = embedding / np.linalg.norm(embedding)
                
                # Calculate cosine similarity
                similarity = np.dot(query_norm, embedding_norm)
                
                # Add to results if above threshold
                if similarity >= min_similarity:
                    results.append((node, float(similarity)))
            
            # Sort and limit results
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:max_results]
        
        # Cache and return results
        self._cache_result(
            "optimized_embedding_search", results, graph_id=id(graph), query_hash=query_hash,
            embedding_attr=embedding_attr, min_similarity=min_similarity, max_results=max_results)
        
        return results
    
    def optimized_subgraph_extraction(self,
                                    graph: nx.Graph,
                                    centers: List[Any],
                                    radius: int = 2,
                                    max_nodes: int = 500) -> nx.Graph:
        """
        Efficiently extract a subgraph around center nodes with optimized performance.
        
        Args:
            graph: NetworkX graph to extract from
            centers: List of center nodes for the subgraph
            radius: Maximum distance from center nodes
            max_nodes: Maximum nodes in the extracted subgraph
            
        Returns:
            nx.Graph: Extracted subgraph
        """
        # Check cache first
        cache_key = (tuple(sorted(centers)), radius, max_nodes)
        cached_result = self._get_cached_result(
            "optimized_subgraph_extraction", graph_id=id(graph), cache_key=cache_key)
        
        if cached_result is not None:
            return cached_result
        
        # Start with center nodes
        included_nodes = set(centers)
        frontier = set(centers)
        current_radius = 0
        
        # Expand outward up to radius
        while frontier and current_radius < radius and len(included_nodes) < max_nodes:
            next_frontier = set()
            
            # Process frontier nodes in parallel for large graphs
            if self.use_parallel and len(frontier) > 100:
                def process_frontier_nodes(nodes):
                    new_nodes = set()
                    for node in nodes:
                        for neighbor in graph.neighbors(node):
                            if neighbor not in included_nodes:
                                new_nodes.add(neighbor)
                    return new_nodes
                
                # Split frontier into batches
                frontier_list = list(frontier)
                batch_size = max(1, len(frontier_list) // self.max_workers)
                batches = [frontier_list[i:i+batch_size] for i in range(0, len(frontier_list), batch_size)]
                
                # Process in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(process_frontier_nodes, batch) for batch in batches]
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            batch_nodes = future.result()
                            next_frontier.update(batch_nodes)
                        except Exception as e:
                            logger.error(f"Error processing frontier batch: {e}")
            else:
                # Sequential processing for smaller frontiers
                for node in frontier:
                    for neighbor in graph.neighbors(node):
                        if neighbor not in included_nodes:
                            next_frontier.add(neighbor)
            
            # Update included nodes and check max_nodes limit
            remaining_capacity = max_nodes - len(included_nodes)
            if len(next_frontier) > remaining_capacity:
                # If we can't include all neighbors, prioritize based on degree
                next_frontier_list = sorted(
                    next_frontier, 
                    key=lambda n: graph.degree(n), 
                    reverse=True
                )
                next_frontier = set(next_frontier_list[:remaining_capacity])
            
            included_nodes.update(next_frontier)
            frontier = next_frontier
            current_radius += 1
        
        # Extract the subgraph
        subgraph = graph.subgraph(included_nodes).copy()
        
        # Cache and return results
        self._cache_result(
            "optimized_subgraph_extraction", subgraph, graph_id=id(graph), cache_key=cache_key)
        
        return subgraph
    
    def attribute_index_search(self,
                              graph: nx.Graph,
                              attribute: str,
                              value: Any) -> List[Any]:
        """
        Efficiently find nodes with a specific attribute value using indices.
        
        Args:
            graph: NetworkX graph to search
            attribute: Attribute name to search
            value: Attribute value to match
            
        Returns:
            List[Any]: List of matching nodes
        """
        # Use index if available
        if attribute in self.attribute_indices:
            index = self.attribute_indices[attribute]
            if value in index:
                return index[value]
            return []
        
        # Fallback to linear search if no index
        results = []
        for node, data in graph.nodes(data=True):
            if attribute in data and data[attribute] == value:
                results.append(node)
        
        return results
