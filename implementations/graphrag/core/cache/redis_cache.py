"""
Redis Graph Cache for GraphRAG.

This module provides a Redis-based caching mechanism for NetworkX graphs,
enabling efficient storage and retrieval of graph components.
"""

import json
import pickle
import hashlib
import networkx as nx
from typing import Dict, Any, Optional, List, Set, Tuple, Union
import redis

from limnos.implementations.graphrag.core.models.entity import Entity
from limnos.implementations.graphrag.core.models.relationship import Relationship
from limnos.implementations.graphrag.core.models.document import DocumentReference


class RedisGraphCache:
    """
    Redis-based cache for NetworkX graphs used in GraphRAG.
    
    This class provides mechanisms to cache entire graphs or subgraphs,
    with support for serialization/deserialization, expiration policies,
    and cache invalidation strategies.
    """
    
    def __init__(
        self, 
        host: str = 'localhost', 
        port: int = 6379, 
        db: int = 0, 
        password: Optional[str] = None,
        ttl: int = 3600,  # Default TTL: 1 hour
        prefix: str = 'graphrag:'
    ):
        """
        Initialize the Redis graph cache.
        
        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            password: Redis password, if required
            ttl: Default time-to-live in seconds for cached items
            prefix: Key prefix for all cached items
        """
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False  # We need binary responses for pickle
        )
        self.ttl = ttl
        self.prefix = prefix
        
    def _generate_key(self, key_components: Union[str, List[str], Dict[str, Any]]) -> str:
        """
        Generate a deterministic Redis key from components.
        
        Args:
            key_components: String, list, or dictionary to use for key generation
            
        Returns:
            A deterministic string key
        """
        if isinstance(key_components, str):
            key_base = key_components
        else:
            # Convert to a consistent string representation
            key_base = json.dumps(key_components, sort_keys=True)
            
        # Create a hash for the key base
        key_hash = hashlib.md5(key_base.encode('utf-8')).hexdigest()
        return f"{self.prefix}{key_hash}"
    
    def cache_graph(self, graph: nx.Graph, key_components: Union[str, List[str], Dict[str, Any]], ttl: Optional[int] = None) -> bool:
        """
        Cache a NetworkX graph in Redis.
        
        Args:
            graph: The NetworkX graph to cache
            key_components: Components to generate a unique key
            ttl: Optional custom TTL, defaults to instance TTL
            
        Returns:
            Boolean indicating success
        """
        key = self._generate_key(key_components)
        ttl = ttl if ttl is not None else self.ttl
        
        try:
            # Serialize the graph
            graph_data = pickle.dumps(graph)
            
            # Store in Redis with expiration
            self.redis_client.set(key, graph_data, ex=ttl)
            return True
        except Exception as e:
            # Log the error in a production environment
            print(f"Error caching graph: {e}")
            return False
    
    def get_graph(self, key_components: Union[str, List[str], Dict[str, Any]]) -> Optional[nx.Graph]:
        """
        Retrieve a cached NetworkX graph from Redis.
        
        Args:
            key_components: Components to generate the unique key
            
        Returns:
            The cached NetworkX graph or None if not found
        """
        key = self._generate_key(key_components)
        
        try:
            graph_data = self.redis_client.get(key)
            if graph_data:
                return pickle.loads(graph_data)
            return None
        except Exception as e:
            # Log the error in a production environment
            print(f"Error retrieving graph: {e}")
            return None
    
    def cache_subgraph(
        self, 
        subgraph: nx.Graph, 
        parent_graph_id: str,
        entity_ids: Optional[List[str]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a subgraph with reference to its parent graph.
        
        Args:
            subgraph: The NetworkX subgraph to cache
            parent_graph_id: Identifier of the parent graph
            entity_ids: Optional list of entity IDs included in this subgraph
            ttl: Optional custom TTL, defaults to instance TTL
            
        Returns:
            Boolean indicating success
        """
        key_components = {
            'parent_graph_id': parent_graph_id,
            'entity_ids': entity_ids if entity_ids else []
        }
        return self.cache_graph(subgraph, key_components, ttl)
    
    def get_subgraph(
        self, 
        parent_graph_id: str,
        entity_ids: Optional[List[str]] = None
    ) -> Optional[nx.Graph]:
        """
        Retrieve a cached subgraph by parent graph ID and optional entity IDs.
        
        Args:
            parent_graph_id: Identifier of the parent graph
            entity_ids: Optional list of entity IDs to include
            
        Returns:
            The cached subgraph or None if not found
        """
        key_components = {
            'parent_graph_id': parent_graph_id,
            'entity_ids': entity_ids if entity_ids else []
        }
        return self.get_graph(key_components)
    
    def invalidate_graph(self, key_components: Union[str, List[str], Dict[str, Any]]) -> bool:
        """
        Invalidate (delete) a cached graph.
        
        Args:
            key_components: Components to generate the unique key
            
        Returns:
            Boolean indicating success
        """
        key = self._generate_key(key_components)
        
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            # Log the error in a production environment
            print(f"Error invalidating graph: {e}")
            return False
    
    def invalidate_document_graphs(self, document_id: str) -> int:
        """
        Invalidate all cached graphs related to a specific document.
        
        Args:
            document_id: The document ID to invalidate caches for
            
        Returns:
            Number of invalidated cache entries
        """
        pattern = f"{self.prefix}*{document_id}*"
        keys = self.redis_client.keys(pattern)
        
        if not keys:
            return 0
            
        return self.redis_client.delete(*keys)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        pattern = f"{self.prefix}*"
        keys = self.redis_client.keys(pattern)
        
        stats = {
            'total_cached_items': len(keys),
            'memory_usage_bytes': 0,
            'key_types': {}
        }
        
        # Get memory usage if available
        try:
            info = self.redis_client.info('memory')
            stats['memory_usage_bytes'] = info.get('used_memory', 0)
        except:
            pass
            
        # Count key types
        for key in keys:
            key_type = self.redis_client.type(key).decode('utf-8')
            stats['key_types'][key_type] = stats['key_types'].get(key_type, 0) + 1
            
        return stats
