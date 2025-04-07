"""
Graph Traversal

Implements core graph traversal algorithms for knowledge graph exploration.
"""

import logging
from typing import Dict, List, Set, Any, Optional, Callable, Tuple, Iterator, Union
from collections import deque
import networkx as nx

logger = logging.getLogger(__name__)

class GraphTraversal:
    """
    Provides graph traversal algorithms for exploring knowledge graphs.
    
    Implements breadth-first, depth-first, and other traversal strategies
    for efficient graph exploration with customizable filters and limits.
    """
    
    def __init__(self, max_depth: int = 5, max_nodes: int = 1000):
        """
        Initialize the graph traversal with configurable limits.
        
        Args:
            max_depth: Maximum traversal depth (default: 5)
            max_nodes: Maximum nodes to visit (default: 1000)
        """
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        logger.info(f"Initialized GraphTraversal with max_depth={max_depth}, max_nodes={max_nodes}")
    
    def breadth_first_search(self, 
                            graph: nx.Graph, 
                            start_node: Any, 
                            node_filter: Optional[Callable[[Any], bool]] = None,
                            edge_filter: Optional[Callable[[Any, Any, Dict], bool]] = None,
                            max_depth: Optional[int] = None, 
                            max_nodes: Optional[int] = None) -> Iterator[Tuple[Any, int, List]]:
        """
        Perform breadth-first search traversal starting from a node.
        
        Args:
            graph: NetworkX graph to traverse
            start_node: Starting node for traversal
            node_filter: Optional function to filter nodes (returns True to include)
            edge_filter: Optional function to filter edges (returns True to traverse)
            max_depth: Maximum traversal depth (overrides instance value if provided)
            max_nodes: Maximum nodes to visit (overrides instance value if provided)
            
        Yields:
            Tuple[Any, int, List]: (node, depth, path) for each visited node
        """
        if start_node not in graph:
            logger.warning(f"Start node {start_node} not found in graph")
            return
        
        # Use instance defaults if not provided
        max_depth = max_depth if max_depth is not None else self.max_depth
        max_nodes = max_nodes if max_nodes is not None else self.max_nodes
        
        # Initialize traversal
        visited = {start_node}
        queue = deque([(start_node, 0, [start_node])])  # (node, depth, path)
        nodes_visited = 1
        
        # Apply node filter if provided
        if node_filter and not node_filter(start_node):
            return
        
        # Yield the start node
        yield start_node, 0, [start_node]
        
        while queue and nodes_visited < max_nodes:
            current_node, depth, path = queue.popleft()
            
            # Stop if max depth reached
            if depth >= max_depth:
                continue
            
            # Get neighbors
            for neighbor in graph.neighbors(current_node):
                # Skip already visited nodes
                if neighbor in visited:
                    continue
                
                # Apply edge filter if provided
                if edge_filter:
                    edge_data = graph.get_edge_data(current_node, neighbor)
                    if not edge_filter(current_node, neighbor, edge_data):
                        continue
                
                # Apply node filter if provided
                if node_filter and not node_filter(neighbor):
                    continue
                
                # Mark as visited and add to queue
                visited.add(neighbor)
                new_path = path + [neighbor]
                queue.append((neighbor, depth + 1, new_path))
                
                # Yield the neighbor
                yield neighbor, depth + 1, new_path
                
                # Update count
                nodes_visited += 1
                if nodes_visited >= max_nodes:
                    break
    
    def depth_first_search(self, 
                          graph: nx.Graph, 
                          start_node: Any, 
                          node_filter: Optional[Callable[[Any], bool]] = None,
                          edge_filter: Optional[Callable[[Any, Any, Dict], bool]] = None,
                          max_depth: Optional[int] = None, 
                          max_nodes: Optional[int] = None) -> Iterator[Tuple[Any, int, List]]:
        """
        Perform depth-first search traversal starting from a node.
        
        Args:
            graph: NetworkX graph to traverse
            start_node: Starting node for traversal
            node_filter: Optional function to filter nodes (returns True to include)
            edge_filter: Optional function to filter edges (returns True to traverse)
            max_depth: Maximum traversal depth (overrides instance value if provided)
            max_nodes: Maximum nodes to visit (overrides instance value if provided)
            
        Yields:
            Tuple[Any, int, List]: (node, depth, path) for each visited node
        """
        if start_node not in graph:
            logger.warning(f"Start node {start_node} not found in graph")
            return
        
        # Use instance defaults if not provided
        max_depth = max_depth if max_depth is not None else self.max_depth
        max_nodes = max_nodes if max_nodes is not None else self.max_nodes
        
        # Initialize traversal
        visited = {start_node}
        stack = [(start_node, 0, [start_node])]  # (node, depth, path)
        nodes_visited = 1
        
        # Apply node filter if provided
        if node_filter and not node_filter(start_node):
            return
        
        # Yield the start node
        yield start_node, 0, [start_node]
        
        while stack and nodes_visited < max_nodes:
            current_node, depth, path = stack.pop()
            
            # Stop if max depth reached
            if depth >= max_depth:
                continue
            
            # Get neighbors in reverse to maintain expected DFS order
            neighbors = list(graph.neighbors(current_node))
            for neighbor in reversed(neighbors):
                # Skip already visited nodes
                if neighbor in visited:
                    continue
                
                # Apply edge filter if provided
                if edge_filter:
                    edge_data = graph.get_edge_data(current_node, neighbor)
                    if not edge_filter(current_node, neighbor, edge_data):
                        continue
                
                # Apply node filter if provided
                if node_filter and not node_filter(neighbor):
                    continue
                
                # Mark as visited and add to stack
                visited.add(neighbor)
                new_path = path + [neighbor]
                stack.append((neighbor, depth + 1, new_path))
                
                # Yield the neighbor
                yield neighbor, depth + 1, new_path
                
                # Update count
                nodes_visited += 1
                if nodes_visited >= max_nodes:
                    break
    
    def bidirectional_search(self,
                            graph: nx.Graph,
                            source: Any,
                            target: Any,
                            max_depth: Optional[int] = None) -> Optional[List]:
        """
        Perform bidirectional search between source and target nodes.
        
        Args:
            graph: NetworkX graph to search
            source: Source node
            target: Target node
            max_depth: Maximum search depth (overrides instance value if provided)
            
        Returns:
            Optional[List]: Path between source and target, or None if no path exists
        """
        if source not in graph or target not in graph:
            logger.warning(f"Source or target node not found in graph")
            return None
        
        # If source and target are the same, return trivial path
        if source == target:
            return [source]
        
        # Use instance default if not provided
        max_depth = max_depth if max_depth is not None else self.max_depth
        
        # Initialize forward and backward search
        forward_visited = {source: [source]}  # node -> path from source
        backward_visited = {target: [target]}  # node -> path from target
        
        # Search queues (node, depth)
        forward_queue = deque([(source, 0)])
        backward_queue = deque([(target, 0)])
        
        # Track intersection
        intersection = None
        
        # Main search loop
        while forward_queue and backward_queue:
            # Check if we've exceeded max depth
            if forward_queue[0][1] + backward_queue[0][1] > max_depth:
                logger.debug(f"Bidirectional search exceeded max depth {max_depth}")
                break
            
            # Forward search step
            if forward_queue:
                intersection = self._bidirectional_step(
                    graph, forward_queue, forward_visited, backward_visited, True)
                if intersection:
                    # Construct the path
                    forward_path = forward_visited[intersection]
                    backward_path = backward_visited[intersection]
                    return forward_path + backward_path[1:][::-1]
            
            # Backward search step
            if backward_queue:
                intersection = self._bidirectional_step(
                    graph, backward_queue, backward_visited, forward_visited, False)
                if intersection:
                    # Construct the path
                    forward_path = forward_visited[intersection]
                    backward_path = backward_visited[intersection]
                    return forward_path + backward_path[1:][::-1]
        
        logger.debug(f"No path found between {source} and {target}")
        return None
    
    def _bidirectional_step(self,
                          graph: nx.Graph,
                          queue: deque,
                          visited: Dict[Any, List],
                          other_visited: Dict[Any, List],
                          is_forward: bool) -> Optional[Any]:
        """
        Perform one step of bidirectional search.
        
        Args:
            graph: NetworkX graph
            queue: Current direction's queue
            visited: Current direction's visited dict
            other_visited: Other direction's visited dict
            is_forward: True if forward step, False if backward
            
        Returns:
            Optional[Any]: Intersection node if found, None otherwise
        """
        current, depth = queue.popleft()
        
        for neighbor in graph.neighbors(current):
            # Skip already visited nodes
            if neighbor in visited:
                continue
            
            # Create path
            new_path = visited[current] + [neighbor]
            visited[neighbor] = new_path
            
            # Check if we found an intersection
            if neighbor in other_visited:
                return neighbor
            
            # Add to queue for next round
            queue.append((neighbor, depth + 1))
        
        return None
    
    def limited_traversal(self,
                         graph: nx.Graph,
                         start_nodes: List[Any],
                         node_importance: Dict[Any, float],
                         max_nodes: Optional[int] = None,
                         alpha: float = 0.5) -> List[Any]:
        """
        Perform traversal with node importance-based expansion limits.
        
        This traversal prioritizes important nodes and limits exploration
        of less important branches.
        
        Args:
            graph: NetworkX graph to traverse
            start_nodes: List of starting nodes
            node_importance: Dict mapping nodes to importance scores (0-1)
            max_nodes: Maximum nodes to include (overrides instance value if provided)
            alpha: Importance weight factor (0-1)
            
        Returns:
            List[Any]: List of visited nodes in priority order
        """
        max_nodes = max_nodes if max_nodes is not None else self.max_nodes
        
        # Initialize with start nodes
        visited_nodes = set(start_nodes)
        frontier = set(start_nodes)
        result_nodes = list(start_nodes)
        
        # Continue until we reach max_nodes or no more frontier
        while frontier and len(result_nodes) < max_nodes:
            # Select highest importance node from frontier
            current = max(frontier, key=lambda n: node_importance.get(n, 0))
            frontier.remove(current)
            
            # Process neighbors
            expansion_limit = int(max(1, alpha * node_importance.get(current, 0) * 10))
            neighbors = list(graph.neighbors(current))
            
            # Sort neighbors by importance
            neighbors.sort(key=lambda n: node_importance.get(n, 0), reverse=True)
            
            # Add top neighbors to frontier
            added = 0
            for neighbor in neighbors:
                if neighbor not in visited_nodes and added < expansion_limit:
                    visited_nodes.add(neighbor)
                    frontier.add(neighbor)
                    result_nodes.append(neighbor)
                    added += 1
                    
                    # Check if we've reached the max nodes
                    if len(result_nodes) >= max_nodes:
                        break
        
        return result_nodes
    
    def multi_source_bfs(self,
                       graph: nx.Graph,
                       source_nodes: List[Any],
                       max_depth: Optional[int] = None,
                       max_nodes: Optional[int] = None) -> Dict[Any, Tuple[int, List]]:
        """
        Perform breadth-first search from multiple source nodes simultaneously.
        
        Args:
            graph: NetworkX graph to traverse
            source_nodes: List of source nodes
            max_depth: Maximum traversal depth (overrides instance value if provided)
            max_nodes: Maximum nodes to visit (overrides instance value if provided)
            
        Returns:
            Dict[Any, Tuple[int, List]]: Map of node -> (min_depth, shortest_path)
        """
        # Use instance defaults if not provided
        max_depth = max_depth if max_depth is not None else self.max_depth
        max_nodes = max_nodes if max_nodes is not None else self.max_nodes
        
        # Filter out source nodes not in graph
        valid_sources = [node for node in source_nodes if node in graph]
        if not valid_sources:
            logger.warning("No valid source nodes found in graph")
            return {}
        
        # Initialize traversal
        results = {}
        visited = set(valid_sources)
        queue = deque([(node, 0, [node]) for node in valid_sources])  # (node, depth, path)
        nodes_visited = len(valid_sources)
        
        # Add source nodes to results
        for node in valid_sources:
            results[node] = (0, [node])
        
        while queue and nodes_visited < max_nodes:
            current_node, depth, path = queue.popleft()
            
            # Stop if max depth reached
            if depth >= max_depth:
                continue
            
            # Get neighbors
            for neighbor in graph.neighbors(current_node):
                # Skip already visited nodes
                if neighbor in visited:
                    continue
                
                # Mark as visited and add to queue
                visited.add(neighbor)
                new_path = path + [neighbor]
                queue.append((neighbor, depth + 1, new_path))
                
                # Add to results
                results[neighbor] = (depth + 1, new_path)
                
                # Update count
                nodes_visited += 1
                if nodes_visited >= max_nodes:
                    break
        
        return results
