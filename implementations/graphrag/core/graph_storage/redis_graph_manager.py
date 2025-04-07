"""
Redis Graph Manager

Manages connections and operations for storing graph data in Redis.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import redis
import networkx as nx

logger = logging.getLogger(__name__)

class RedisGraphManager:
    """
    Manager for storing and retrieving graph data in Redis.
    
    This class handles the connection to Redis and provides methods for
    storing, retrieving, and managing graph data.
    """
    redis_client: redis.Redis
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 6379, 
                 db: int = 0, 
                 prefix: str = "graphrag",
                 password: Optional[str] = None,
                 ttl: int = 86400 * 7):  # Default TTL: 7 days
        """
        Initialize the Redis Graph Manager.
        
        Args:
            host: Redis host address
            port: Redis port
            db: Redis database number
            prefix: Prefix for all keys stored by this manager
            password: Redis password, if required
            ttl: Time-to-live for cached items in seconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.prefix = prefix
        self.password = password
        self.ttl = ttl
        
        # Initialize Redis client
        self._initialize_redis_client()
        
        logger.info(f"Initialized RedisGraphManager with prefix '{prefix}' on {host}:{port}/{db}")
    
    def _initialize_redis_client(self) -> None:
        """Initialize the Redis client connection"""
        try:
            self.redis_client: redis.Redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False  # We want bytes for binary data
            )
            self.redis_client.ping()  # Test connection
            logger.info("Successfully connected to Redis")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def graph_key(self, graph_id: str) -> str:
        """Generate a Redis key for a graph"""
        return f"{self.prefix}:graph:{graph_id}"
    
    def metadata_key(self, graph_id: str) -> str:
        """Generate a Redis key for graph metadata"""
        return f"{self.prefix}:graph:{graph_id}:metadata"
    
    def store_graph(self, graph_id: str, graph: nx.Graph, metadata: Optional[Dict[Union[str, bytes], Union[str, int, float, bytes]]] = None) -> bool:
        """
        Store a graph and its metadata in Redis.
        
        Args:
            graph_id: Unique identifier for the graph
            graph: NetworkX graph object
            metadata: Optional metadata dict to store with the graph
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert NetworkX graph to a serialized format (using GraphSerializer)
            from .graph_serializer import GraphSerializer
            serializer = GraphSerializer()
            graph_data = serializer.serialize(graph)
            
            # Store the graph data
            graph_key = self.graph_key(graph_id)
            self.redis_client.set(graph_key, graph_data)
            self.redis_client.expire(graph_key, self.ttl)
            
            # Store metadata if provided
            if metadata:
                metadata_key = self.metadata_key(graph_id)
                # Use explicit typing that matches Redis's expectations
                # Redis expects Mapping[str | bytes, bytes | float | int | str]
                typed_metadata: Dict[Union[str, bytes], Union[str, int, float, bytes]] = {}
                
                # Convert keys and values to compatible types
                for k, v in metadata.items():
                    # Ensure keys are str or bytes
                    key = k if isinstance(k, (str, bytes)) else str(k)
                    # Ensure values are valid Redis types
                    value = v if isinstance(v, (str, int, float, bytes)) else str(v)
                    typed_metadata[key] = value
                    
                self.redis_client.hset(metadata_key, mapping=typed_metadata)
                self.redis_client.expire(metadata_key, self.ttl)
            
            logger.info(f"Stored graph '{graph_id}' in Redis")
            return True
        except Exception as e:
            logger.error(f"Error storing graph '{graph_id}' in Redis: {e}")
            return False
    
    def retrieve_graph(self, graph_id: str) -> Tuple[Optional[nx.Graph], Optional[Dict]]:
        """
        Retrieve a graph and its metadata from Redis.
        
        Args:
            graph_id: Unique identifier for the graph
            
        Returns:
            Tuple[Optional[nx.Graph], Optional[Dict]]: The graph and its metadata, or None if not found
        """
        try:
            # Get the graph data
            graph_key = self.graph_key(graph_id)
            graph_data = self.redis_client.get(graph_key)
            
            if not graph_data:
                logger.warning(f"Graph '{graph_id}' not found in Redis")
                return None, None
            
            # Deserialize the graph
            from .graph_serializer import GraphSerializer
            serializer = GraphSerializer()
            graph = serializer.deserialize(graph_data)
            
            # Get metadata if available
            metadata = None
            metadata_key = self.metadata_key(graph_id)
            if self.redis_client.exists(metadata_key):
                metadata = self.redis_client.hgetall(metadata_key)
                # Convert bytes keys/values to strings
                metadata = {k.decode('utf-8') if isinstance(k, bytes) else k: 
                           v.decode('utf-8') if isinstance(v, bytes) else v 
                           for k, v in metadata.items()}
            
            logger.info(f"Retrieved graph '{graph_id}' from Redis")
            return graph, metadata
        except Exception as e:
            logger.error(f"Error retrieving graph '{graph_id}' from Redis: {e}")
            return None, None
    
    def delete_graph(self, graph_id: str) -> bool:
        """
        Delete a graph and its metadata from Redis.
        
        Args:
            graph_id: Unique identifier for the graph
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Delete the graph and metadata
            graph_key = self.graph_key(graph_id)
            metadata_key = self.metadata_key(graph_id)
            
            keys_to_delete = [graph_key, metadata_key]
            self.redis_client.delete(*keys_to_delete)
            
            logger.info(f"Deleted graph '{graph_id}' from Redis")
            return True
        except Exception as e:
            logger.error(f"Error deleting graph '{graph_id}' from Redis: {e}")
            return False
    
    def list_graphs(self, pattern: str = "*") -> List[str]:
        """
        List all graph IDs in Redis matching the pattern.
        
        Args:
            pattern: Pattern to match graph IDs (default: all graphs)
            
        Returns:
            List[str]: List of matching graph IDs
        """
        try:
            # Get all matching graph keys
            prefix_pattern = f"{self.prefix}:graph:{pattern}"
            matching_keys = self.redis_client.keys(prefix_pattern)
            
            # Extract graph IDs from keys
            graph_ids = []
            for key in matching_keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                if ":metadata" not in key_str:  # Skip metadata keys
                    graph_id = key_str.split(f"{self.prefix}:graph:")[1]
                    graph_ids.append(graph_id)
            
            return graph_ids
        except Exception as e:
            logger.error(f"Error listing graphs in Redis: {e}")
            return []
    
    def flush_cache(self) -> bool:
        """
        Flush all graphs and metadata from Redis for this prefix.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get all keys with our prefix
            keys_to_delete = self.redis_client.keys(f"{self.prefix}:*")
            
            if keys_to_delete:
                self.redis_client.delete(*keys_to_delete)
                logger.info(f"Flushed {len(keys_to_delete)} keys with prefix '{self.prefix}' from Redis")
            else:
                logger.info(f"No keys with prefix '{self.prefix}' found in Redis")
            
            return True
        except Exception as e:
            logger.error(f"Error flushing Redis cache: {e}")
            return False
    
    @classmethod
    def from_environment(cls) -> 'RedisGraphManager':
        """
        Create a RedisGraphManager instance from environment variables.
        
        Environment variables:
            GRAPHRAG_REDIS_HOST: Redis host (default: localhost)
            GRAPHRAG_REDIS_PORT: Redis port (default: 6379)
            GRAPHRAG_REDIS_DB: Redis database number (default: 0)
            GRAPHRAG_REDIS_PREFIX: Key prefix (default: graphrag)
            GRAPHRAG_REDIS_PASSWORD: Redis password (default: None)
            GRAPHRAG_REDIS_TTL: Time-to-live in seconds (default: 604800 - 7 days)
        
        Returns:
            RedisGraphManager: Initialized manager
        """
        host = os.environ.get("GRAPHRAG_REDIS_HOST", "localhost")
        port = int(os.environ.get("GRAPHRAG_REDIS_PORT", 6379))
        db = int(os.environ.get("GRAPHRAG_REDIS_DB", 0))
        prefix = os.environ.get("GRAPHRAG_REDIS_PREFIX", "graphrag")
        password = os.environ.get("GRAPHRAG_REDIS_PASSWORD", None)
        ttl = int(os.environ.get("GRAPHRAG_REDIS_TTL", 86400 * 7))
        
        return cls(host=host, port=port, db=db, prefix=prefix, password=password, ttl=ttl)
