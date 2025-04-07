"""
Graph Cache Manager

Provides caching mechanisms for frequently accessed graphs to improve performance.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List, Set, Tuple
import networkx as nx
from collections import OrderedDict

logger = logging.getLogger(__name__)

class LRUCache:
    """
    Least Recently Used (LRU) cache implementation for storing graphs in memory.
    """
    
    def __init__(self, capacity: int = 100):
        """
        Initialize the LRU cache with a specified capacity.
        
        Args:
            capacity: Maximum number of items to store in the cache
        """
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.RLock()  # Reentrant lock for thread safety
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache, moving it to the front if found.
        
        Args:
            key: The cache key
            
        Returns:
            Any: The cached value or None if not found
        """
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key: str, value: Any) -> None:
        """
        Add an item to the cache, evicting least recently used items if needed.
        
        Args:
            key: The cache key
            value: The value to cache
        """
        with self.lock:
            if key in self.cache:
                # Remove existing item
                self.cache.pop(key)
            
            # Add to end (most recently used)
            self.cache[key] = value
            
            # Evict least recently used if over capacity
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
    
    def remove(self, key: str) -> bool:
        """
        Remove an item from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            bool: True if the item was removed, False if not found
        """
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        with self.lock:
            self.cache.clear()
    
    def contains(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        with self.lock:
            return key in self.cache
    
    def size(self) -> int:
        """Get the current cache size."""
        with self.lock:
            return len(self.cache)
    
    def keys(self) -> List[str]:
        """Get all keys in the cache."""
        with self.lock:
            return list(self.cache.keys())


class GraphCacheManager:
    """
    Manages in-memory caching of graph data for high-performance access.
    
    This class provides an LRU caching mechanism for frequently accessed graphs,
    reducing the need to retrieve them from Redis.
    """
    
    def __init__(self, 
                 capacity: int = 100, 
                 ttl: int = 3600,
                 auto_refresh: bool = True,
                 refresh_threshold: float = 0.75):
        """
        Initialize the Graph Cache Manager.
        
        Args:
            capacity: Maximum number of graphs to cache in memory
            ttl: Time-to-live for cached graphs in seconds
            auto_refresh: Whether to automatically refresh TTL on access
            refresh_threshold: Threshold ratio (0-1) of TTL to trigger refresh
        """
        self.capacity = capacity
        self.ttl = ttl
        self.auto_refresh = auto_refresh
        self.refresh_threshold = refresh_threshold
        
        # Initialize LRU cache for graph objects
        self.graph_cache = LRUCache(capacity)
        
        # Track expiration times
        self.expiration_times: Dict[str, float] = {}
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'insertions': 0,
            'refreshes': 0
        }
        
        # Set up cache maintenance
        self.maintenance_interval = 60  # Run maintenance every 60 seconds
        self.maintenance_thread = None
        self.shutdown_flag = threading.Event()
        
        # Start maintenance thread
        self._start_maintenance_thread()
        
        logger.info(f"Initialized GraphCacheManager with capacity {capacity}, TTL {ttl}s")
    
    def _start_maintenance_thread(self) -> None:
        """Start the cache maintenance thread."""
        if self.maintenance_thread is None:
            self.maintenance_thread = threading.Thread(
                target=self._maintenance_loop,
                daemon=True  # Make thread daemon so it exits when main thread exits
            )
            self.maintenance_thread.start()
            logger.info("Started cache maintenance thread")
    
    def _maintenance_loop(self) -> None:
        """Maintenance loop for expired cache entries."""
        while not self.shutdown_flag.is_set():
            self._evict_expired()
            self.shutdown_flag.wait(self.maintenance_interval)
    
    def _evict_expired(self) -> None:
        """Evict expired entries from the cache."""
        current_time = time.time()
        expired_keys = []
        
        # Find expired keys
        for key, expiration_time in list(self.expiration_times.items()):
            if current_time > expiration_time:
                expired_keys.append(key)
        
        # Remove expired items
        for key in expired_keys:
            if self.graph_cache.remove(key):
                del self.expiration_times[key]
                self.stats['evictions'] += 1
        
        if expired_keys:
            logger.debug(f"Evicted {len(expired_keys)} expired items from cache")
    
    def get(self, graph_id: str) -> Optional[nx.Graph]:
        """
        Get a graph from the cache.
        
        Args:
            graph_id: Unique identifier for the graph
            
        Returns:
            Optional[nx.Graph]: The cached graph or None if not found
        """
        graph = self.graph_cache.get(graph_id)
        
        if graph is not None:
            # Cache hit
            self.stats['hits'] += 1
            
            # Update expiration time if auto-refresh is enabled
            if self.auto_refresh:
                current_time = time.time()
                expiration_time = self.expiration_times.get(graph_id, 0)
                time_left = expiration_time - current_time
                
                # Refresh if below threshold
                if time_left < (self.ttl * self.refresh_threshold):
                    self.expiration_times[graph_id] = current_time + self.ttl
                    self.stats['refreshes'] += 1
        else:
            # Cache miss
            self.stats['misses'] += 1
        
        return graph
    
    def put(self, graph_id: str, graph: nx.Graph) -> None:
        """
        Add a graph to the cache.
        
        Args:
            graph_id: Unique identifier for the graph
            graph: NetworkX graph object to cache
        """
        self.graph_cache.put(graph_id, graph)
        self.expiration_times[graph_id] = time.time() + self.ttl
        self.stats['insertions'] += 1
    
    def remove(self, graph_id: str) -> bool:
        """
        Remove a graph from the cache.
        
        Args:
            graph_id: Unique identifier for the graph
            
        Returns:
            bool: True if removed, False if not found
        """
        if self.graph_cache.remove(graph_id):
            if graph_id in self.expiration_times:
                del self.expiration_times[graph_id]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cached graphs."""
        self.graph_cache.clear()
        self.expiration_times.clear()
        logger.info("Cleared graph cache")
    
    def contains(self, graph_id: str) -> bool:
        """
        Check if a graph is in the cache.
        
        Args:
            graph_id: Unique identifier for the graph
            
        Returns:
            bool: True if cached, False otherwise
        """
        return self.graph_cache.contains(graph_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        stats = dict(self.stats)
        stats['size'] = self.graph_cache.size()
        stats['capacity'] = self.capacity
        
        if stats['hits'] + stats['misses'] > 0:
            stats['hit_ratio'] = stats['hits'] / (stats['hits'] + stats['misses'])
        else:
            stats['hit_ratio'] = 0.0
        
        return stats
    
    def shutdown(self) -> None:
        """Shut down the cache manager and stop maintenance."""
        self.shutdown_flag.set()
        if self.maintenance_thread:
            self.maintenance_thread.join(timeout=1.0)
            self.maintenance_thread = None
        logger.info("Shut down cache manager")
