"""
Path Representation Structures for PathRAG

This module provides specialized data structures for representing, indexing and
querying paths in the PathRAG implementation. These structures are optimized
for efficient retrieval operations.
"""

import numpy as np
import json
import pickle
import os
from pathlib import Path as FilePath
from typing import List, Dict, Any, Set, Tuple, Optional, Union, Callable, Iterable, Type
import logging
import networkx as nx
from collections import defaultdict
import heapq
import time
import sys  # For sys.getsizeof in get_stats
from dataclasses import dataclass, field, asdict

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PathNode:
    """
    Represents a node in a path with all its metadata.
    """
    id: str
    text: str
    type: str
    document_id: str = ""
    document_title: str = ""
    position: int = -1  # Position in the path
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PathNode':
        """Create from dictionary representation."""
        metadata = {k: v for k, v in data.items() 
                    if k not in ['id', 'text', 'type', 'document_id', 
                               'document_title', 'position', 'metadata']}
        
        if 'metadata' in data:
            metadata.update(data['metadata'])
            
        return cls(
            id=data['id'],
            text=data['text'],
            type=data['type'],
            document_id=data.get('document_id', ''),
            document_title=data.get('document_title', ''),
            position=data.get('position', -1),
            metadata=metadata
        )

@dataclass
class PathEdge:
    """
    Represents an edge in a path with all its metadata.
    """
    id: str
    source_id: str
    target_id: str
    type: str
    weight: float = 1.0
    confidence: float = 0.0
    document_id: str = ""
    document_title: str = ""
    position: int = -1  # Position in the path
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PathEdge':
        """Create from dictionary representation."""
        metadata = {k: v for k, v in data.items() 
                    if k not in ['id', 'source_id', 'target_id', 'type', 'weight', 
                               'confidence', 'document_id', 'document_title', 
                               'position', 'metadata']}
        
        if 'metadata' in data:
            metadata.update(data['metadata'])
            
        return cls(
            id=data['id'],
            source_id=data['source_id'],
            target_id=data['target_id'],
            type=data['type'],
            weight=data.get('weight', 1.0),
            confidence=data.get('confidence', 0.0),
            document_id=data.get('document_id', ''),
            document_title=data.get('document_title', ''),
            position=data.get('position', -1),
            metadata=metadata
        )

@dataclass
class Path:
    """
    Represents a complete path with its nodes, edges, and metadata.
    """
    id: str
    nodes: List[PathNode]
    edges: List[PathEdge]
    score: float = 0.0
    length: int = 0
    source_query: str = ""
    creation_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set length if not explicitly provided."""
        if self.length == 0:
            self.length = len(self.edges)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges],
            'score': self.score,
            'length': self.length,
            'source_query': self.source_query,
            'creation_time': self.creation_time,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Path':
        """Create from dictionary representation."""
        return cls(
            id=data['id'],
            nodes=[PathNode.from_dict(node_data) for node_data in data['nodes']],
            edges=[PathEdge.from_dict(edge_data) for edge_data in data['edges']],
            score=data.get('score', 0.0),
            length=data.get('length', 0),
            source_query=data.get('source_query', ''),
            creation_time=data.get('creation_time', time.time()),
            metadata=data.get('metadata', {})
        )
    
    def get_node_texts(self) -> List[str]:
        """Get the text of all nodes in the path."""
        return [node.text for node in self.nodes]
    
    def get_edge_types(self) -> List[str]:
        """Get the types of all edges in the path."""
        return [edge.type for edge in self.edges]
    
    def get_document_ids(self) -> Set[str]:
        """Get all unique document IDs referenced in the path."""
        doc_ids = set()
        for node in self.nodes:
            if node.document_id:
                doc_ids.add(node.document_id)
        for edge in self.edges:
            if edge.document_id:
                doc_ids.add(edge.document_id)
        return doc_ids
    
    def contains_entity(self, entity_text: str, case_sensitive: bool = False) -> bool:
        """Check if the path contains a specific entity text."""
        if case_sensitive:
            return any(node.text == entity_text for node in self.nodes)
        else:
            entity_text_lower = entity_text.lower()
            return any(node.text.lower() == entity_text_lower for node in self.nodes)
    
    def contains_relationship(self, relationship_type: str) -> bool:
        """Check if the path contains a specific relationship type."""
        return any(edge.type == relationship_type for edge in self.edges)
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert the path to a NetworkX directed graph."""
        G: nx.DiGraph = nx.DiGraph()
        
        # Add nodes
        for node in self.nodes:
            G.add_node(node.id, **node.to_dict())
        
        # Add edges
        for edge in self.edges:
            G.add_edge(edge.source_id, edge.target_id, **edge.to_dict())
        
        return G


class PathIndex:
    """
    Efficient index structure for paths, optimized for retrieval operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the path index with configuration.
        
        Args:
            config: Configuration dictionary with options for the index
        """
        self.config = config or {}
        
        # Path storage
        self.paths: Dict[str, Path] = {}
        
        # Indexes for efficient retrieval
        self.entity_to_paths: Dict[str, Set[str]] = defaultdict(set)  # entity text -> path IDs
        self.entity_type_to_paths: Dict[str, Set[str]] = defaultdict(set)  # entity type -> path IDs
        self.relationship_to_paths: Dict[str, Set[str]] = defaultdict(set)  # relationship type -> path IDs
        self.document_to_paths: Dict[str, Set[str]] = defaultdict(set)  # document ID -> path IDs
        
        # Term-based index for text search
        self.term_to_paths: Dict[str, Set[str]] = defaultdict(set)  # term -> path IDs
        
        # Path length index
        self.length_to_paths: Dict[int, Set[str]] = defaultdict(set)  # path length -> path IDs
        
        # Path score index (for top-k retrieval)
        self.scored_paths: List[Tuple[float, str]] = []  # (score, path_id) for heap operations
        
        # Storage directory
        self.storage_dir = self.config.get('storage_dir', None)
    
    def add_path(self, path: Path) -> None:
        """
        Add a path to the index.
        
        Args:
            path: Path object to add
        """
        # Skip if already indexed
        if path.id in self.paths:
            return
        
        # Add to path storage
        self.paths[path.id] = path
        
        # Index by entities
        for node in path.nodes:
            self.entity_to_paths[node.text.lower()].add(path.id)
            self.entity_type_to_paths[node.type].add(path.id)
            
            # Index terms for text search
            for term in self._extract_terms(node.text):
                self.term_to_paths[term].add(path.id)
        
        # Index by relationships
        for edge in path.edges:
            self.relationship_to_paths[edge.type].add(path.id)
        
        # Index by documents
        for doc_id in path.get_document_ids():
            self.document_to_paths[doc_id].add(path.id)
        
        # Index by length
        self.length_to_paths[path.length].add(path.id)
        
        # Index by score
        heapq.heappush(self.scored_paths, (-path.score, path.id))  # Negative for max-heap
    
    def add_paths(self, paths: List[Path]) -> None:
        """
        Add multiple paths to the index.
        
        Args:
            paths: List of Path objects to add
        """
        for path in paths:
            self.add_path(path)
    
    def remove_path(self, path_id: str) -> bool:
        """
        Remove a path from the index.
        
        Args:
            path_id: ID of the path to remove
            
        Returns:
            True if path was removed, False if not found
        """
        if path_id not in self.paths:
            return False
        
        path = self.paths[path_id]
        
        # Remove from path storage
        del self.paths[path_id]
        
        # Remove from entity index
        for node in path.nodes:
            if path_id in self.entity_to_paths[node.text.lower()]:
                self.entity_to_paths[node.text.lower()].remove(path_id)
            
            if path_id in self.entity_type_to_paths[node.type]:
                self.entity_type_to_paths[node.type].remove(path_id)
            
            # Remove from term index
            for term in self._extract_terms(node.text):
                if path_id in self.term_to_paths[term]:
                    self.term_to_paths[term].remove(path_id)
        
        # Remove from relationship index
        for edge in path.edges:
            if path_id in self.relationship_to_paths[edge.type]:
                self.relationship_to_paths[edge.type].remove(path_id)
        
        # Remove from document index
        for doc_id in path.get_document_ids():
            if path_id in self.document_to_paths[doc_id]:
                self.document_to_paths[doc_id].remove(path_id)
        
        # Remove from length index
        if path_id in self.length_to_paths[path.length]:
            self.length_to_paths[path.length].remove(path_id)
        
        # Remove from score index (rebuild heap)
        self.scored_paths = [item for item in self.scored_paths if item[1] != path_id]
        heapq.heapify(self.scored_paths)
        
        return True
    
    def get_path(self, path_id: str) -> Optional[Path]:
        """
        Get a path by its ID.
        
        Args:
            path_id: ID of the path to retrieve
            
        Returns:
            Path object or None if not found
        """
        return self.paths.get(path_id)
    
    def get_paths_by_entity(self, entity_text: str, case_sensitive: bool = False) -> List[Path]:
        """
        Get paths containing a specific entity.
        
        Args:
            entity_text: Text of the entity to search for
            case_sensitive: Whether to perform case-sensitive matching
            
        Returns:
            List of Path objects
        """
        if not case_sensitive:
            entity_text = entity_text.lower()
            path_ids = self.entity_to_paths.get(entity_text, set())
        else:
            # For case-sensitive search, we need to check all paths
            path_ids = {path_id for path_id, path in self.paths.items() 
                       if path.contains_entity(entity_text, case_sensitive=True)}
        
        return [self.paths[path_id] for path_id in path_ids]
    
    def get_paths_by_entity_type(self, entity_type: str) -> List[Path]:
        """
        Get paths containing entities of a specific type.
        
        Args:
            entity_type: Type of entities to search for
            
        Returns:
            List of Path objects
        """
        path_ids = self.entity_type_to_paths.get(entity_type, set())
        return [self.paths[path_id] for path_id in path_ids]
    
    def get_paths_by_relationship(self, relationship_type: str) -> List[Path]:
        """
        Get paths containing a specific relationship type.
        
        Args:
            relationship_type: Type of relationship to search for
            
        Returns:
            List of Path objects
        """
        path_ids = self.relationship_to_paths.get(relationship_type, set())
        return [self.paths[path_id] for path_id in path_ids]
    
    def get_paths_by_document(self, document_id: str) -> List[Path]:
        """
        Get paths associated with a specific document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of Path objects
        """
        path_ids = self.document_to_paths.get(document_id, set())
        return [self.paths[path_id] for path_id in path_ids]
    
    def get_paths_by_length(self, length: int) -> List[Path]:
        """
        Get paths of a specific length.
        
        Args:
            length: Length of paths to retrieve
            
        Returns:
            List of Path objects
        """
        path_ids = self.length_to_paths.get(length, set())
        return [self.paths[path_id] for path_id in path_ids]
    
    def get_paths_by_length_range(self, min_length: int, max_length: int) -> List[Path]:
        """
        Get paths within a length range.
        
        Args:
            min_length: Minimum path length (inclusive)
            max_length: Maximum path length (inclusive)
            
        Returns:
            List of Path objects
        """
        path_ids = set()
        for length in range(min_length, max_length + 1):
            path_ids.update(self.length_to_paths.get(length, set()))
        
        return [self.paths[path_id] for path_id in path_ids]
    
    def get_top_paths(self, k: int = 10) -> List[Path]:
        """
        Get the top-k highest scoring paths.
        
        Args:
            k: Number of paths to retrieve
            
        Returns:
            List of Path objects
        """
        # Use heapq to get top-k efficiently
        top_k = heapq.nsmallest(k, self.scored_paths)  # Smallest because we use negative scores
        return [self.paths[path_id] for _, path_id in top_k]
    
    def search_paths(self, query: str, k: int = 10) -> List[Tuple[Path, float]]:
        """
        Search for paths matching a text query.
        
        Args:
            query: Text query to search for
            k: Maximum number of results to return
            
        Returns:
            List of (Path, score) tuples
        """
        # Extract terms from query
        query_terms = self._extract_terms(query)
        
        # Score each path based on term overlap
        path_scores: Dict[str, float] = defaultdict(float)
        
        for term in query_terms:
            matching_paths = self.term_to_paths.get(term, set())
            
            for path_id in matching_paths:
                # Simple TF-IDF-like scoring
                # Term frequency: how many terms from the query are in the path
                # Inverse document frequency: penalize common terms
                path_scores[path_id] += 1.0 / (1.0 + len(matching_paths))
        
        # Sort by score
        scored_paths = [(score, path_id) for path_id, score in path_scores.items()]
        scored_paths.sort(reverse=True)
        
        # Return top-k
        return [(self.paths[path_id], score) for score, path_id in scored_paths[:k]]
    
    def filter_paths(self, filter_func: Callable[[Path], bool]) -> List[Path]:
        """
        Filter paths using a custom function.
        
        Args:
            filter_func: Function that takes a Path and returns a boolean
            
        Returns:
            List of Path objects that match the filter
        """
        return [path for path in self.paths.values() if filter_func(path)]
    
    def combine_filters(self, *conditions: Set[str]) -> Set[str]:
        """
        Combine multiple filter conditions with AND logic.
        
        Args:
            *conditions: Sets of path IDs from different filter conditions
            
        Returns:
            Set of path IDs that match all conditions
        """
        if not conditions:
            return set()
        
        # Start with all paths from first condition
        result = conditions[0]
        
        # Intersect with each additional condition
        for condition in conditions[1:]:
            result = result.intersection(condition)
        
        return result
    
    def advanced_search(self, 
                        entity_texts: Optional[List[str]] = None,
                        entity_types: Optional[List[str]] = None,
                        relationship_types: Optional[List[str]] = None,
                        document_ids: Optional[List[str]] = None,
                        min_length: Optional[int] = None,
                        max_length: Optional[int] = None,
                        min_score: Optional[float] = None,
                        k: int = 10) -> List[Path]:
        """
        Advanced search with multiple criteria.
        
        Args:
            entity_texts: List of entity texts to filter by
            entity_types: List of entity types to filter by
            relationship_types: List of relationship types to filter by
            document_ids: List of document IDs to filter by
            min_length: Minimum path length
            max_length: Maximum path length
            min_score: Minimum path score
            k: Maximum number of results to return
            
        Returns:
            List of Path objects matching all criteria
        """
        filter_conditions = []
        
        # Entity text filter
        if entity_texts:
            entity_path_ids = set()
            for entity_text in entity_texts:
                entity_path_ids.update(self.entity_to_paths.get(entity_text.lower(), set()))
            filter_conditions.append(entity_path_ids)
        
        # Entity type filter
        if entity_types:
            entity_type_path_ids = set()
            for entity_type in entity_types:
                entity_type_path_ids.update(self.entity_type_to_paths.get(entity_type, set()))
            filter_conditions.append(entity_type_path_ids)
        
        # Relationship type filter
        if relationship_types:
            relationship_path_ids = set()
            for relationship_type in relationship_types:
                relationship_path_ids.update(self.relationship_to_paths.get(relationship_type, set()))
            filter_conditions.append(relationship_path_ids)
        
        # Document ID filter
        if document_ids:
            document_path_ids = set()
            for document_id in document_ids:
                document_path_ids.update(self.document_to_paths.get(document_id, set()))
            filter_conditions.append(document_path_ids)
        
        # Length filter
        if min_length is not None or max_length is not None:
            min_len = min_length if min_length is not None else 1
            max_len = max_length if max_length is not None else float('inf')
            
            length_path_ids = set()
            for length, path_ids in self.length_to_paths.items():
                if min_len <= length <= max_len:
                    length_path_ids.update(path_ids)
            
            filter_conditions.append(length_path_ids)
        
        # Combine all filters
        if filter_conditions:
            result_path_ids = self.combine_filters(*filter_conditions)
        else:
            # If no filters, use all paths
            result_path_ids = set(self.paths.keys())
        
        # Score filter
        if min_score is not None:
            result_path_ids = {
                path_id for path_id in result_path_ids 
                if self.paths[path_id].score >= min_score
            }
        
        # Sort by score and return top-k
        result_paths = [(self.paths[path_id], self.paths[path_id].score) 
                       for path_id in result_path_ids]
        result_paths.sort(key=lambda x: x[1], reverse=True)
        
        return [path for path, _ in result_paths[:k]]
    
    def _extract_terms(self, text: str) -> List[str]:
        """
        Extract searchable terms from text.
        
        Args:
            text: Text to extract terms from
            
        Returns:
            List of normalized terms
        """
        # Simple whitespace tokenization and lowercasing
        # In a production system, you'd want more sophisticated tokenization,
        # stemming, etc.
        if not text:
            return []
        
        return [term.lower() for term in text.split() if term]
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the index to a file.
        
        Args:
            filepath: Path to save to (if None, uses storage_dir/paths_index.pkl)
            
        Returns:
            Path to the saved file
        """
        if filepath is None:
            if self.storage_dir is None:
                raise ValueError("No storage directory specified")
            
            # Ensure directory exists
            os.makedirs(self.storage_dir, exist_ok=True)
            filepath = os.path.join(self.storage_dir, "paths_index.pkl")
        
        # Convert paths to dictionaries for serialization
        serialized_data = {
            'paths': {path_id: path.to_dict() for path_id, path in self.paths.items()},
            'entity_to_paths': dict(self.entity_to_paths),
            'entity_type_to_paths': dict(self.entity_type_to_paths),
            'relationship_to_paths': dict(self.relationship_to_paths),
            'document_to_paths': dict(self.document_to_paths),
            'term_to_paths': dict(self.term_to_paths),
            'length_to_paths': dict(self.length_to_paths),
            'scored_paths': self.scored_paths
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(serialized_data, f)
        
        logger.info(f"Saved path index to {filepath}")
        return filepath
    
    def load(self, filepath: str) -> 'PathIndex':
        """
        Load the index from a file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            Loaded PathIndex
        """
        with open(filepath, 'rb') as f:
            serialized_data = pickle.load(f)
        
        # Deserialize paths
        self.paths = {
            path_id: Path.from_dict(path_data) 
            for path_id, path_data in serialized_data['paths'].items()
        }
        
        # Load indexes
        self.entity_to_paths = defaultdict(set, serialized_data['entity_to_paths'])
        self.entity_type_to_paths = defaultdict(set, serialized_data['entity_type_to_paths'])
        self.relationship_to_paths = defaultdict(set, serialized_data['relationship_to_paths'])
        self.document_to_paths = defaultdict(set, serialized_data['document_to_paths'])
        self.term_to_paths = defaultdict(set, serialized_data['term_to_paths'])
        self.length_to_paths = defaultdict(set, serialized_data['length_to_paths'])
        self.scored_paths = serialized_data['scored_paths']
        
        logger.info(f"Loaded path index from {filepath}")
        return self
    
    def export_paths(self, filepath: str, format: str = 'json') -> str:
        """
        Export all paths to a file.
        
        Args:
            filepath: Path to export to
            format: Export format ('json' or 'jsonl')
            
        Returns:
            Path to the exported file
        """
        # Ensure directory exists
        FilePath(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            # Export all paths as a JSON array
            with open(filepath, 'w', encoding='utf-8') as f:
                paths_data = [path.to_dict() for path in self.paths.values()]
                json.dump(paths_data, f, indent=2)
                
        elif format == 'jsonl':
            # Export each path as a separate JSON line
            with open(filepath, 'w', encoding='utf-8') as f:
                for path in self.paths.values():
                    f.write(json.dumps(path.to_dict()) + '\n')
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(self.paths)} paths to {filepath}")
        return filepath
    
    def import_paths(self, filepath: str, format: str = 'json') -> int:
        """
        Import paths from a file.
        
        Args:
            filepath: Path to import from
            format: Import format ('json' or 'jsonl')
            
        Returns:
            Number of imported paths
        """
        if format == 'json':
            # Import paths from a JSON array
            with open(filepath, 'r', encoding='utf-8') as f:
                paths_data = json.load(f)
                
            for path_data in paths_data:
                path = Path.from_dict(path_data)
                self.add_path(path)
                
            return len(paths_data)
            
        elif format == 'jsonl':
            # Import paths from separate JSON lines
            count = 0
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    path_data = json.loads(line.strip())
                    path = Path.from_dict(path_data)
                    self.add_path(path)
                    count += 1
            
            return count
        else:
            raise ValueError(f"Unsupported import format: {format}")
    
    def clear(self) -> None:
        """Clear all data from the index."""
        self.paths.clear()
        self.entity_to_paths.clear()
        self.entity_type_to_paths.clear()
        self.relationship_to_paths.clear()
        self.document_to_paths.clear()
        self.term_to_paths.clear()
        self.length_to_paths.clear()
        self.scored_paths.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'path_count': len(self.paths),
            'entity_count': len(self.entity_to_paths),
            'entity_type_count': len(self.entity_type_to_paths),
            'relationship_type_count': len(self.relationship_to_paths),
            'document_count': len(self.document_to_paths),
            'term_count': len(self.term_to_paths),
            'average_path_length': sum(path.length for path in self.paths.values()) / max(1, len(self.paths)),
            'max_path_length': max([path.length for path in self.paths.values()]) if self.paths else 0,
            'min_path_length': min([path.length for path in self.paths.values()]) if self.paths else 0,
            'average_path_score': sum(path.score for path in self.paths.values()) / max(1, len(self.paths))
        }
