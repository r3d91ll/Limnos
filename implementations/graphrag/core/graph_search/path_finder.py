"""
Path Finder

Implements path finding algorithms for knowledge graph navigation.
"""

import logging
import heapq
from typing import Dict, List, Set, Any, Optional, Callable, Tuple, Iterator, Union, Mapping
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

class PathFinder:
    """
    Provides algorithms for finding optimal paths in knowledge graphs.
    
    Implements various path finding algorithms including shortest path,
    weighted path, top-k paths, and semantic path finding based on
    node and edge attributes.
    """
    
    def __init__(self, max_paths: int = 10, max_path_length: int = 10):
        """
        Initialize the path finder with configurable limits.
        
        Args:
            max_paths: Maximum number of paths to return in multi-path search
            max_path_length: Maximum path length to consider
        """
        self.max_paths = max_paths
        self.max_path_length = max_path_length
        logger.info(f"Initialized PathFinder with max_paths={max_paths}, max_path_length={max_path_length}")
    
    def shortest_path(self,
                     graph: nx.Graph,
                     source: Any,
                     target: Any,
                     weight: Optional[str] = None) -> Optional[List]:
        """
        Find the shortest path between source and target nodes.
        
        Args:
            graph: NetworkX graph to search
            source: Source node
            target: Target node
            weight: Optional edge attribute to use as weight
            
        Returns:
            Optional[List]: Shortest path from source to target, or None if no path exists
        """
        try:
            path = nx.shortest_path(graph, source=source, target=target, weight=weight)
            logger.debug(f"Found shortest path of length {len(path)} between {source} and {target}")
            return path
        except nx.NetworkXNoPath:
            logger.debug(f"No path found between {source} and {target}")
            return None
        except nx.NetworkXError as e:
            logger.warning(f"Error finding shortest path: {e}")
            return None
    
    def all_shortest_paths(self,
                          graph: nx.Graph,
                          source: Any,
                          target: Any,
                          weight: Optional[str] = None,
                          max_paths: Optional[int] = None) -> List[List]:
        """
        Find all shortest paths between source and target nodes.
        
        Args:
            graph: NetworkX graph to search
            source: Source node
            target: Target node
            weight: Optional edge attribute to use as weight
            max_paths: Maximum number of paths to return
            
        Returns:
            List[List]: List of all shortest paths, empty if no path exists
        """
        max_paths = max_paths if max_paths is not None else self.max_paths
        
        try:
            paths = list(nx.all_shortest_paths(graph, source=source, target=target, weight=weight))
            if max_paths and len(paths) > max_paths:
                paths = paths[:max_paths]
            
            logger.debug(f"Found {len(paths)} shortest paths between {source} and {target}")
            return paths
        except nx.NetworkXNoPath:
            logger.debug(f"No path found between {source} and {target}")
            return []
        except nx.NetworkXError as e:
            logger.warning(f"Error finding all shortest paths: {e}")
            return []
    
    def k_shortest_paths(self,
                        graph: nx.Graph,
                        source: Any,
                        target: Any,
                        k: Optional[int] = None,
                        weight: str = 'weight') -> List[Tuple[List, float]]:
        """
        Find k shortest loopless paths between source and target nodes.
        
        Uses a variant of Yen's algorithm for k shortest loopless paths.
        
        Args:
            graph: NetworkX graph to search
            source: Source node
            target: Target node
            k: Number of shortest paths to find
            weight: Edge attribute to use as weight
            
        Returns:
            List[Tuple[List, float]]: List of (path, path_length) tuples
        """
        k = k if k is not None else self.max_paths
        
        # Return empty list if source or target not in graph
        if source not in graph or target not in graph:
            logger.warning(f"Source or target node not found in graph")
            return []
        
        # Ensure weights exist by using 1.0 as default if not specified
        for u, v, d in graph.edges(data=True):
            if weight not in d:
                d[weight] = 1.0
        
        # Find the shortest path
        try:
            shortest_path = nx.shortest_path(graph, source=source, target=target, weight=weight)
            shortest_path_length = nx.shortest_path_length(graph, source=source, target=target, weight=weight)
        except nx.NetworkXNoPath:
            logger.debug(f"No path found between {source} and {target}")
            return []
        
        # Initialize with the first shortest path
        A: List[Tuple[List[Any], float]] = [(shortest_path, shortest_path_length)]  # List of (path, length) tuples
        B: List[Tuple[float, List[Any]]] = []  # Heap of (length, path) tuples for potential k-shortest paths
        
        # Find k-1 more paths
        for i in range(1, k):
            # Last path found
            prev_path = A[-1][0]
            
            # For each node in the previous path except the last
            for j in range(len(prev_path) - 1):
                # Spur node is the j-th node in the previous path
                spur_node = prev_path[j]
                
                # Root path is the prefix of the previous path up to the spur node
                root_path = prev_path[:j+1]
                
                # Remove edges that are part of previously found paths with the same root
                removed_edges = []
                for path, _ in A:
                    if len(path) > j and path[:j+1] == root_path:
                        u = path[j]
                        v = path[j+1]
                        if graph.has_edge(u, v):
                            edge_data = graph.get_edge_data(u, v)
                            removed_edges.append((u, v, edge_data))
                            graph.remove_edge(u, v)
                
                # Remove nodes in root path from graph (except spur node)
                removed_nodes = []
                for node in root_path[:-1]:  # All except the spur node
                    neighbors = list(graph.neighbors(node))
                    for neighbor in neighbors:
                        edge_data = graph.get_edge_data(node, neighbor)
                        removed_edges.append((node, neighbor, edge_data))
                    graph.remove_node(node)
                    removed_nodes.append(node)
                
                # Find shortest path from spur node to target
                spur_path = None
                spur_path_length = float('inf')
                try:
                    spur_path = nx.shortest_path(graph, source=spur_node, target=target, weight=weight)
                    spur_path_length = nx.shortest_path_length(graph, source=spur_node, target=target, weight=weight)
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    # No path found or error, skip
                    pass
                
                # Add back removed nodes and edges
                for node in removed_nodes:
                    graph.add_node(node)
                for u, v, edge_data in removed_edges:
                    graph.add_edge(u, v, **edge_data)
                
                # If a new path was found
                if spur_path is not None:
                    # Full path is root_path + spur_path[1:]
                    # (exclude the spur node from the spur path as it's already in the root path)
                    total_path = root_path + spur_path[1:]
                    
                    # Calculate path length
                    total_path_length = 0
                    for idx in range(len(total_path) - 1):
                        u, v = total_path[idx], total_path[idx + 1]
                        edge_data = graph.get_edge_data(u, v)
                        total_path_length += edge_data.get(weight, 1.0)
                    
                    # Add the path to B if it's not already in A or B
                    path_tuple = (total_path_length, tuple(total_path))
                    if path_tuple not in [(l, tuple(p)) for p, l in A] + [(l, tuple(p)) for l, p in B]:
                        heapq.heappush(B, (total_path_length, total_path))
            
            # No more candidates
            if not B:
                break
            
            # Get the next shortest path
            path_length, path = heapq.heappop(B)
            A.append((path, path_length))
        
        logger.debug(f"Found {len(A)} of k={k} shortest paths between {source} and {target}")
        return A
    
    def weighted_paths(self,
                      graph: nx.Graph,
                      source: Any,
                      target: Any,
                      weight_function: Callable[[Any, Any, Mapping[str, Any]], float],
                      max_paths: Optional[int] = None) -> List[Tuple[List[Any], float]]:
        """
        Find paths based on a custom weight function.
        
        This method uses a modified Dijkstra's algorithm with a priority queue
        to find paths based on custom edge weights.
        
        Args:
            graph: NetworkX graph to search
            source: Source node
            target: Target node
            weight_function: Function taking (u, v, edge_data) and returning a weight
            max_paths: Maximum number of paths to return
            
        Returns:
            List[Tuple[List, float]]: List of (path, total_weight) tuples
        """
        max_paths = max_paths if max_paths is not None else self.max_paths
        
        # Return empty list if source or target not in graph
        if source not in graph or target not in graph:
            logger.warning(f"Source or target node not found in graph")
            return []
        
        # Initialize priority queue with (weight, node, path)
        pq: List[Tuple[float, Any, List[Any]]] = [(0, source, [source])]
        visited: Set[Any] = set()
        results: List[Tuple[List[Any], float]] = []
        
        while pq and len(results) < max_paths:
            weight, node, path = heapq.heappop(pq)
            
            # If we found a path to the target
            if node == target:
                results.append((path, weight))
                continue
            
            # Skip if already visited
            if node in visited:
                continue
            
            # Mark as visited
            visited.add(node)
            
            # Process neighbors
            for neighbor in graph.neighbors(node):
                if neighbor in visited:
                    continue
                
                edge_data = graph.get_edge_data(node, neighbor)
                edge_weight = weight_function(node, neighbor, edge_data)
                
                # Add to queue
                heapq.heappush(pq, (weight + edge_weight, neighbor, path + [neighbor]))
        
        logger.debug(f"Found {len(results)} weighted paths between {source} and {target}")
        return results
    
    def semantic_paths(self,
                      graph: nx.Graph,
                      source: Any,
                      target: Any,
                      query_vector: np.ndarray,
                      embedding_attr: str = 'embedding',
                      max_paths: Optional[int] = None) -> List[Tuple[List, float]]:
        """
        Find paths based on semantic similarity to a query vector.
        
        This method uses cosine similarity as a heuristic to guide path finding
        toward semantically relevant nodes.
        
        Args:
            graph: NetworkX graph to search
            source: Source node
            target: Target node
            query_vector: Query embedding vector for similarity calculation
            embedding_attr: Node attribute containing embedding vectors
            max_paths: Maximum number of paths to return
            
        Returns:
            List[Tuple[List, float]]: List of (path, similarity_score) tuples
        """
        max_paths = max_paths if max_paths is not None else self.max_paths
        
        # Return empty list if source or target not in graph
        if source not in graph or target not in graph:
            logger.warning(f"Source or target node not found in graph")
            return []
        
        # Helper function to calculate semantic guidance weight
        def semantic_weight(node: Any) -> float:
            # Get node embedding vector
            node_data = graph.nodes[node]
            if embedding_attr not in node_data:
                return 0.0
            
            # Calculate cosine similarity
            embedding = node_data[embedding_attr]
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Normalize vectors
            embedding = embedding / np.linalg.norm(embedding)
            query_norm = query_vector / np.linalg.norm(query_vector)
            
            # Calculate similarity (1 - cosine similarity for minimization)
            similarity: float = 1.0 - float(np.dot(embedding, query_norm))
            return similarity
        
        # A* search with semantic heuristic
        # Initialize priority queue with (weight, node, path, path_score)
        start_weight = semantic_weight(source)
        pq: List[Tuple[float, float, Any, List[Any], float]] = [(start_weight, 0, source, [source], start_weight)]
        visited: Set[Any] = set()
        results: List[Tuple[List[Any], float]] = []
        
        while pq and len(results) < max_paths:
            # Get node with lowest total weight (f = g + h)
            f, g, node, path, score = heapq.heappop(pq)
            
            # If we found a path to the target
            if node == target:
                # We invert the score for the results to get similarity rather than distance
                results.append((path, 1.0 - score / len(path)))
                continue
            
            # Skip if already visited
            if node in visited:
                continue
            
            # Skip if path is too long
            if len(path) > self.max_path_length:
                continue
            
            # Mark as visited
            visited.add(node)
            
            # Process neighbors
            for neighbor in graph.neighbors(node):
                if neighbor in visited:
                    continue
                
                # Calculate weights
                neighbor_similarity = semantic_weight(neighbor)
                edge_weight = 1.0  # Uniform step cost
                
                # Path-based metrics
                new_g = g + edge_weight
                new_score = score + neighbor_similarity
                new_path = path + [neighbor]
                
                # A* formula: f = g + h (path cost + heuristic)
                # Use average semantic similarity as the score
                avg_score = new_score / len(new_path)
                new_f = new_g + avg_score
                
                # Add to queue
                heapq.heappush(pq, (new_f, new_g, neighbor, new_path, new_score))
        
        logger.debug(f"Found {len(results)} semantic paths between {source} and {target}")
        return results
    
    def diverse_paths(self,
                     graph: nx.Graph,
                     source: Any,
                     target: Any,
                     diversity_threshold: float = 0.3,
                     max_paths: Optional[int] = None) -> List[List[Any]]:
        """
        Find diverse paths between source and target nodes.
        
        This method finds paths that have minimal node and edge overlap
        to provide diverse options for knowledge graph exploration.
        
        Args:
            graph: NetworkX graph to search
            source: Source node
            target: Target node
            diversity_threshold: Minimum Jaccard distance between path sets
            max_paths: Maximum number of paths to return
            
        Returns:
            List[List]: List of diverse paths
        """
        max_paths = max_paths if max_paths is not None else self.max_paths
        
        # Find many candidate paths using k-shortest paths
        path_tuples = self.k_shortest_paths(
            graph, source, target, k=max_paths * 3, weight='weight')
        
        if not path_tuples:
            return []
        
        # Keep only the paths
        candidate_paths: List[List[Any]] = [path for path, _ in path_tuples]
        
        # Calculate path diversity
        selected_paths: List[List[Any]] = [candidate_paths[0]]  # Start with shortest path
        
        for path in candidate_paths[1:]:
            path_set = set(path)
            
            # Check diversity against all selected paths
            is_diverse = True
            for selected_path in selected_paths:
                selected_set = set(selected_path)
                
                # Calculate Jaccard distance (1 - Jaccard similarity)
                intersection = len(path_set.intersection(selected_set))
                union = len(path_set.union(selected_set))
                jaccard_similarity = intersection / union if union > 0 else 0
                jaccard_distance = 1 - jaccard_similarity
                
                if jaccard_distance < diversity_threshold:
                    is_diverse = False
                    break
            
            # Add path if diverse enough
            if is_diverse:
                selected_paths.append(path)
            
            # Stop if we have enough paths
            if len(selected_paths) >= max_paths:
                break
        
        logger.debug(f"Found {len(selected_paths)} diverse paths between {source} and {target}")
        return selected_paths
