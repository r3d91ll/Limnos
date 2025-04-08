"""
Graph Cache Manager for GraphRAG.

This module provides a cache management layer that abstracts the underlying
caching implementation and maintains the separation between universal and 
framework-specific data, following Limnos's architectural principles.
"""

import json
import uuid
import networkx as nx
from typing import Dict, Any, Optional, List, Set, Tuple, Union, Type

from limnos.implementations.graphrag.core.models.entity import Entity
from limnos.implementations.graphrag.core.models.relationship import Relationship
from limnos.implementations.graphrag.core.models.document import DocumentReference
from limnos.implementations.graphrag.core.cache.redis_cache import RedisGraphCache


class GraphCacheManager:
    """
    Manager for graph caching operations in GraphRAG.
    
    This class provides a high-level interface for caching graph components,
    handling serialization, invalidation policies, and maintaining the
    separation between universal and framework-specific metadata.
    """
    
    def __init__(
        self,
        cache_implementation: Optional[RedisGraphCache] = None,
        cache_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the graph cache manager.
        
        Args:
            cache_implementation: Optional cache implementation to use
            cache_config: Configuration for creating a new cache implementation
        """
        if cache_implementation:
            self.cache = cache_implementation
        else:
            config = cache_config or {}
            self.cache = RedisGraphCache(**config)
        
        # Track document-specific cache keys
        # Keys are document IDs, values are lists of cache key dictionaries
        self.document_cache_keys: Dict[str, List[Dict[str, Any]]] = {}
        
    def cache_document_graph(
        self, 
        graph: nx.Graph, 
        document: DocumentReference,
        ttl: Optional[int] = None
    ) -> str:
        """
        Cache a document-specific graph.
        
        Args:
            graph: The NetworkX graph to cache
            document: Reference to the document
            ttl: Optional time-to-live for this graph
            
        Returns:
            Cache key identifier
        """
        # Generate a cache key specific to this document
        # Following Limnos principles - maintain separation of framework-specific metadata
        from datetime import datetime
        
        # Generate a cache key specific to this document using available attributes
        key_components = {
            'document_id': document.document_id,
            'document_version': document.document_id,  # Use document_id as version since version is not available
            'framework': 'graphrag',  # Explicitly framework-specific
            'timestamp': document.last_processed.isoformat() if document.last_processed else datetime.now().isoformat()
        }
        
        # Track this key for document-based invalidation
        if document.document_id not in self.document_cache_keys:
            self.document_cache_keys[document.document_id] = []
        self.document_cache_keys[document.document_id].append(key_components)
        
        # Cache the graph
        success = self.cache.cache_graph(graph, key_components, ttl)
        
        # Return a string representation of the key
        return json.dumps(key_components)
    
    def get_document_graph(
        self, 
        document: DocumentReference
    ) -> Optional[nx.Graph]:
        """
        Retrieve a document-specific graph from cache.
        
        Args:
            document: Reference to the document
            
        Returns:
            The cached graph or None if not found
        """
        from datetime import datetime
        
        key_components = {
            'document_id': document.document_id,
            'document_version': document.document_id,  # Use document_id as version since version is not available
            'framework': 'graphrag',
            'timestamp': document.last_processed.isoformat() if document.last_processed else datetime.now().isoformat()
        }
        
        return self.cache.get_graph(key_components)
    
    def cache_entity_subgraph(
        self,
        subgraph: nx.Graph,
        document: DocumentReference,
        entity: Entity,
        depth: int = 1,
        ttl: Optional[int] = None
    ) -> str:
        """
        Cache a subgraph centered on a specific entity.
        
        Args:
            subgraph: The NetworkX subgraph to cache
            document: Reference to the document
            entity: The central entity of this subgraph
            depth: The traversal depth from the central entity
            ttl: Optional time-to-live
            
        Returns:
            Cache key identifier
        """
        key_components = {
            'document_id': document.document_id,
            'entity_id': entity.id,
            'depth': depth,
            'framework': 'graphrag'
        }
        
        # Track this key for document-based invalidation
        if document.document_id not in self.document_cache_keys:
            self.document_cache_keys[document.document_id] = []
        self.document_cache_keys[document.document_id].append(key_components)
        
        # Cache the subgraph
        success = self.cache.cache_graph(subgraph, key_components, ttl)
        
        # Return a string representation of the key
        return json.dumps(key_components)
    
    def get_entity_subgraph(
        self,
        document: DocumentReference,
        entity: Entity,
        depth: int = 1
    ) -> Optional[nx.Graph]:
        """
        Retrieve an entity-centered subgraph from cache.
        
        Args:
            document: Reference to the document
            entity: The central entity of the subgraph
            depth: The traversal depth from the central entity
            
        Returns:
            The cached subgraph or None if not found
        """
        key_components = {
            'document_id': document.document_id,
            'entity_id': entity.id,
            'depth': depth,
            'framework': 'graphrag'
        }
        
        return self.cache.get_graph(key_components)
    
    def cache_merged_graph(
        self,
        graph: nx.Graph,
        document_ids: List[str],
        graph_id: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> str:
        """
        Cache a merged graph combining multiple documents.
        
        Args:
            graph: The merged NetworkX graph to cache
            document_ids: List of document IDs included in this graph
            graph_id: Optional identifier for this merged graph
            ttl: Optional time-to-live
            
        Returns:
            Cache key identifier
        """
        # Generate a unique ID for this merged graph if not provided
        if not graph_id:
            graph_id = str(uuid.uuid4())
            
        key_components = {
            'merged_graph_id': graph_id,
            'document_ids': sorted(document_ids),  # Sort for consistency
            'framework': 'graphrag'
        }
        
        # Track this key for each document for invalidation
        for doc_id in document_ids:
            if doc_id not in self.document_cache_keys:
                self.document_cache_keys[doc_id] = []
            self.document_cache_keys[doc_id].append(key_components)
        
        # Cache the graph
        success = self.cache.cache_graph(graph, key_components, ttl)
        
        # Return a string representation of the key
        return json.dumps(key_components)
    
    def get_merged_graph(
        self,
        document_ids: List[str],
        graph_id: Optional[str] = None
    ) -> Optional[nx.Graph]:
        """
        Retrieve a merged graph from cache.
        
        Args:
            document_ids: List of document IDs included in this graph
            graph_id: Optional identifier for this merged graph
            
        Returns:
            The cached merged graph or None if not found
        """
        key_components = {
            'merged_graph_id': graph_id if graph_id else '',
            'document_ids': sorted(document_ids),  # Sort for consistency
            'framework': 'graphrag'
        }
        
        return self.cache.get_graph(key_components)
    
    def invalidate_document_caches(self, document_id: str) -> int:
        """
        Invalidate all caches related to a specific document.
        
        Args:
            document_id: The document ID to invalidate caches for
            
        Returns:
            Number of invalidated cache entries
        """
        # First, use the Redis pattern-based invalidation
        count = self.cache.invalidate_document_graphs(document_id)
        
        # Then, explicitly invalidate tracked keys
        if document_id in self.document_cache_keys:
            for key_components in self.document_cache_keys[document_id]:
                self.cache.invalidate_graph(key_components)
                
            # Clear the tracking list
            count += len(self.document_cache_keys[document_id])
            self.document_cache_keys[document_id] = []
            
        return count
    
    def invalidate_entity_caches(self, entity_id: str) -> int:
        """
        Invalidate all caches related to a specific entity.
        
        Args:
            entity_id: The entity ID to invalidate caches for
            
        Returns:
            Number of invalidated cache entries
        """
        # Use the Redis pattern-based invalidation
        pattern = f"{self.cache.prefix}*{entity_id}*"
        keys = self.cache.redis_client.keys(pattern)
        
        if not keys:
            return 0
            
        return self.cache.redis_client.delete(*keys)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        base_stats = self.cache.get_cache_stats()
        
        # Add framework-specific stats
        framework_stats = {
            'tracked_documents': len(self.document_cache_keys),
            'total_tracked_keys': sum(len(keys) for keys in self.document_cache_keys.values())
        }
        
        return {**base_stats, **framework_stats}
    
    def clear_all_caches(self) -> int:
        """
        Clear all GraphRAG caches.
        
        Returns:
            Number of cleared cache entries
        """
        pattern = f"{self.cache.prefix}*"
        keys = self.cache.redis_client.keys(pattern)
        
        if not keys:
            return 0
            
        # Clear Redis caches
        count = self.cache.redis_client.delete(*keys)
        
        # Clear tracking dictionaries
        self.document_cache_keys = {}
        
        return count
