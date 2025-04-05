"""
Path Constructor for PathRAG

This module provides path construction capabilities for the PathRAG implementation.
It builds paths from entities and relationships that form the basis for path-based retrieval.
"""

import networkx as nx
from typing import List, Dict, Any, Set, Tuple, Optional
import logging
from collections import defaultdict
import heapq
import math
import random

# Configure logging
logger = logging.getLogger(__name__)

class PathConstructor:
    """
    Constructs paths from entities and relationships for PathRAG.
    
    This constructor creates paths that form the basis for path-based retrieval,
    optimizing for semantic relevance and information value.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the path constructor with configuration.
        
        Args:
            config: Configuration dictionary with options for path construction
        """
        self.config = config or {}
        
        # Maximum path length (number of edges)
        self.max_path_length = self.config.get("max_path_length", 5)
        
        # Minimum path length (number of edges)
        self.min_path_length = self.config.get("min_path_length", 1)
        
        # Maximum number of paths to generate per query
        self.max_paths_per_query = self.config.get("max_paths_per_query", 20)
        
        # Whether to use relationship confidence in path ranking
        self.use_confidence_scores = self.config.get("use_confidence_scores", True)
        
        # Weight factors for different relationship types
        self.relationship_weights = self.config.get("relationship_weights", {
            "CO_OCCURS_WITH": 0.7,
            "HAS_ATTRIBUTE": 0.8,
            "PART_OF": 0.9,
            "BELONGS_TO": 0.85,
            "LOCATED_IN": 0.8,
            "CREATED_BY": 0.9,
            "USED_FOR": 0.85,
            "RELATED_TO": 0.7,
            "INSTANCE_OF": 0.8,
            "CAUSES": 0.85,
            "PRECEDES": 0.8,
            "FOLLOWED_BY": 0.8,
            "CONTRADICTS": 0.9
        })
        
        # Default weight for relationship types not specified
        self.default_relationship_weight = self.config.get("default_relationship_weight", 0.7)
        
        # Factor to encourage path diversity (higher = more diverse paths)
        self.diversity_factor = self.config.get("diversity_factor", 0.3)
        
        # Graph representation of entities and relationships
        self.graph = None
    
    def build_graph(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> nx.MultiDiGraph:
        """
        Build a graph from entities and relationships.
        
        Args:
            entities: List of entity dictionaries
            relationships: List of relationship dictionaries
            
        Returns:
            NetworkX MultiDiGraph representing the entity-relationship graph
        """
        # Create a directed multigraph (allows multiple edges between same nodes)
        G = nx.MultiDiGraph()
        
        # Add entities as nodes
        for entity in entities:
            G.add_node(
                entity["id"],
                text=entity["text"],
                type=entity["type"],
                metadata=entity
            )
        
        # Add relationships as edges
        for relationship in relationships:
            source_id = relationship["source"]["id"]
            target_id = relationship["target"]["id"]
            
            # Skip if source or target is not in the graph
            if source_id not in G or target_id not in G:
                continue
                
            # Calculate edge weight based on relationship type and confidence
            weight = self._calculate_edge_weight(relationship)
            
            G.add_edge(
                source_id,
                target_id,
                id=relationship["id"],
                type=relationship["type"],
                weight=weight,
                metadata=relationship
            )
        
        self.graph = G
        return G
    
    def _calculate_edge_weight(self, relationship: Dict[str, Any]) -> float:
        """
        Calculate edge weight for a relationship.
        
        Args:
            relationship: Relationship dictionary
            
        Returns:
            Edge weight value (lower is better for shortest paths)
        """
        # Get relationship type weight
        rel_type = relationship["type"]
        type_weight = self.relationship_weights.get(rel_type, self.default_relationship_weight)
        
        # Get confidence score
        confidence = relationship.get("confidence", 0.5)
        
        if self.use_confidence_scores:
            # Combine type weight and confidence (higher confidence = lower edge weight)
            weight = 1.0 - (type_weight * confidence)
        else:
            # Use only type weight
            weight = 1.0 - type_weight
        
        # Ensure weight is positive
        return max(0.01, weight)
    
    def construct_paths(
        self, query_entities: List[Dict[str, Any]], graph: Optional[nx.MultiDiGraph] = None
    ) -> List[Dict[str, Any]]:
        """
        Construct paths starting from query entities.
        
        Args:
            query_entities: List of entities extracted from the query
            graph: Optional graph to use (uses self.graph if not provided)
            
        Returns:
            List of path dictionaries with metadata
        """
        G = graph or self.graph
        if G is None:
            logger.warning("No graph available for path construction")
            return []
        
        # Extract entity IDs from query entities
        query_entity_ids = [entity["id"] for entity in query_entities if entity["id"] in G]
        
        if not query_entity_ids:
            logger.warning("No valid query entities found in the graph")
            return []
        
        all_paths = []
        
        # For each query entity, find paths
        for start_entity_id in query_entity_ids:
            # Use shortest path algorithms to find relevant paths
            paths = self._find_paths_from_entity(G, start_entity_id)
            all_paths.extend(paths)
        
        # Rank and filter paths
        ranked_paths = self._rank_paths(all_paths)
        
        # Limit to maximum number of paths
        return ranked_paths[:self.max_paths_per_query]
    
    def _find_paths_from_entity(self, G: nx.MultiDiGraph, start_entity_id: str) -> List[Dict[str, Any]]:
        """
        Find paths starting from a specific entity.
        
        Args:
            G: NetworkX graph of entities and relationships
            start_entity_id: Starting entity ID
            
        Returns:
            List of path dictionaries
        """
        paths = []
        
        # Use simple BFS for paths up to max_path_length
        # More sophisticated approaches (e.g., shortest path algorithms) could be used here
        visited = {start_entity_id: 0}  # node: distance from start
        queue = [(start_entity_id, [start_entity_id])]  # (node, path)
        
        while queue:
            current, path = queue.pop(0)
            
            # If path is at minimum length, add it to the result
            if len(path) - 1 >= self.min_path_length:
                path_info = self._create_path_dict(G, path)
                paths.append(path_info)
            
            # If we've reached max path length, don't explore further
            if visited[current] >= self.max_path_length:
                continue
            
            # Explore neighbors
            for neighbor in G.neighbors(current):
                # Skip if already in the path to avoid cycles
                if neighbor in path:
                    continue
                
                # Add neighbor to visited and queue
                if neighbor not in visited or visited[neighbor] > visited[current] + 1:
                    visited[neighbor] = visited[current] + 1
                    queue.append((neighbor, path + [neighbor]))
        
        return paths
    
    def _create_path_dict(self, G: nx.MultiDiGraph, node_path: List[str]) -> Dict[str, Any]:
        """
        Create a path dictionary from a list of node IDs.
        
        Args:
            G: NetworkX graph
            node_path: List of node IDs forming a path
            
        Returns:
            Path dictionary with metadata
        """
        # Extract edges along the path
        edges = []
        for i in range(len(node_path) - 1):
            source = node_path[i]
            target = node_path[i + 1]
            
            # Get edge data (might be multiple edges)
            edge_data = G.get_edge_data(source, target)
            
            # If multiple edges exist, take the one with lowest weight
            if edge_data:
                # Sort by weight (lowest first) and take the first
                edge_key = min(edge_data.keys(), key=lambda k: edge_data[k]["weight"])
                edge = edge_data[edge_key]
                edges.append({
                    "source": source,
                    "target": target,
                    "id": edge.get("id", f"edge_{source}_{target}"),
                    "type": edge.get("type", "RELATED_TO"),
                    "weight": edge.get("weight", 1.0),
                    "metadata": edge.get("metadata", {})
                })
        
        # Calculate path score based on edge weights and path length
        total_weight = sum(edge["weight"] for edge in edges)
        path_score = math.exp(-total_weight) / math.sqrt(len(edges))
        
        # Create full path dictionary
        path_dict = {
            "id": f"path_{'_'.join(node_path)}",
            "nodes": node_path,
            "edges": edges,
            "length": len(edges),
            "score": path_score,
            "node_data": [G.nodes[node] for node in node_path]
        }
        
        return path_dict
    
    def _rank_paths(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank paths by relevance and diversity.
        
        Args:
            paths: List of path dictionaries
            
        Returns:
            Ranked list of path dictionaries
        """
        if not paths:
            return []
            
        # First, sort by score (descending)
        sorted_paths = sorted(paths, key=lambda p: p["score"], reverse=True)
        
        # If diversity is not important, return sorted paths
        if self.diversity_factor <= 0:
            return sorted_paths
        
        # Apply maximum marginal relevance for diversity
        ranked_paths = []
        remaining_paths = sorted_paths.copy()
        
        # Add the highest scoring path first
        ranked_paths.append(remaining_paths.pop(0))
        
        # Add remaining paths based on combination of score and diversity
        while remaining_paths and len(ranked_paths) < self.max_paths_per_query:
            best_idx = -1
            best_score = float("-inf")
            
            for i, path in enumerate(remaining_paths):
                # Original relevance score
                relevance = path["score"]
                
                # Calculate diversity penalty (similarity to already selected paths)
                max_similarity = max(self._path_similarity(path, selected) 
                                   for selected in ranked_paths)
                
                # Combined score (higher is better)
                combined_score = (1 - self.diversity_factor) * relevance - self.diversity_factor * max_similarity
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = i
            
            if best_idx >= 0:
                ranked_paths.append(remaining_paths.pop(best_idx))
            else:
                break
        
        return ranked_paths
    
    def _path_similarity(self, path1: Dict[str, Any], path2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two paths.
        
        Args:
            path1: First path dictionary
            path2: Second path dictionary
            
        Returns:
            Similarity score (0-1, higher means more similar)
        """
        # Use Jaccard similarity of nodes
        nodes1 = set(path1["nodes"])
        nodes2 = set(path2["nodes"])
        
        if not nodes1 or not nodes2:
            return 0.0
            
        intersection = len(nodes1.intersection(nodes2))
        union = len(nodes1.union(nodes2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_graph(self) -> Optional[nx.MultiDiGraph]:
        """
        Get the current graph.
        
        Returns:
            NetworkX MultiDiGraph or None if not built
        """
        return self.graph
