#!/usr/bin/env python3
"""
PathRAG Redis Integration

This module integrates Redis with PathRAG as a high-performance caching solution,
replacing the previous RAMDisk implementation. It provides a seamless interface
for storing and retrieving vectors and metadata in Redis.

With 256GB of RAM available, this implementation can significantly improve
performance by keeping vectors in memory, mitigating PCIe bottlenecks.
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Set

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Redis cache
from src.redis_cache import RedisCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "logs", "pathrag_redis.log"))
    ]
)
logger = logging.getLogger(__name__)

class PathRAGRedisAdapter:
    """
    Adapter class to integrate Redis with PathRAG
    
    This class provides a drop-in replacement for the PathRAG vector storage
    and retrieval functionality, using Redis as the backend instead of RAMDisk.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        vector_dim: int = 384,
        prefix: str = "pathrag",
        ttl: int = 86400 * 7,  # 7 days default TTL
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the PathRAG Redis adapter
        
        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            vector_dim: Dimension of vectors to store
            prefix: Prefix for all keys in Redis
            ttl: Time-to-live for cached items in seconds
            cache_dir: Directory to use for fallback disk cache if Redis is unavailable
        """
        self.vector_dim = vector_dim
        self.prefix = prefix
        self.ttl = ttl
        self.cache_dir = cache_dir
        
        # Initialize Redis cache
        try:
            self.redis_cache = RedisCache(
                host=host,
                port=port,
                db=db,
                password=password,
                vector_dim=vector_dim,
                prefix=prefix,
                ttl=ttl
            )
            self.redis_available = True
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            self.redis_available = False
            logger.error(f"Failed to initialize Redis cache: {e}")
            logger.info("Falling back to disk cache")
            
            # Set up disk cache directory if Redis is unavailable
            if cache_dir is None:
                self.cache_dir = os.path.join(project_root, "cache")
            
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Using disk cache at {self.cache_dir}")
    
    def store_vector(
        self, 
        path: str, 
        vector: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a vector in the cache
        
        Args:
            path: Unique path identifier for the vector
            vector: The vector to store
            metadata: Optional metadata to store with the vector
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.redis_available:
            return self.redis_cache.store_vector(path, vector, metadata)
        else:
            return self._store_vector_disk(path, vector, metadata)
    
    def get_vector(self, path: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve a vector from the cache
        
        Args:
            path: Path identifier for the vector
            
        Returns:
            Tuple of (vector, metadata) or None if not found
        """
        if self.redis_available:
            return self.redis_cache.get_vector(path)
        else:
            return self._get_vector_disk(path)
    
    def search_similar_vectors(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors in the cache
        
        Args:
            query_vector: The query vector
            top_k: Number of results to return
            
        Returns:
            List of tuples (path, score, metadata)
        """
        if self.redis_available:
            return self.redis_cache.search_similar_vectors(query_vector, top_k)
        else:
            return self._search_similar_vectors_disk(query_vector, top_k)
    
    def delete_vector(self, path: str) -> bool:
        """
        Delete a vector from the cache
        
        Args:
            path: Path identifier for the vector
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.redis_available:
            return self.redis_cache.delete_vector(path)
        else:
            return self._delete_vector_disk(path)
    
    def get_all_paths(self) -> List[str]:
        """
        Get all paths stored in the cache
        
        Returns:
            List of paths
        """
        if self.redis_available:
            return self.redis_cache.get_all_paths()
        else:
            return self._get_all_paths_disk()
    
    def clear_cache(self) -> bool:
        """
        Clear all cached vectors
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.redis_available:
            return self.redis_cache.clear_cache()
        else:
            return self._clear_cache_disk()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dict with cache statistics
        """
        if self.redis_available:
            return self.redis_cache.get_stats()
        else:
            return self._get_stats_disk()
    
    def _store_vector_disk(
        self, 
        path: str, 
        vector: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Fallback disk storage implementation"""
        try:
            # Create a safe filename from the path
            filename = self._path_to_filename(path)
            filepath = os.path.join(self.cache_dir, filename)
            
            # Save vector and metadata
            data = {
                "vector": vector.tolist(),
                "metadata": metadata or {},
                "path": path,
                "timestamp": time.time()
            }
            
            with open(filepath, "w") as f:
                json.dump(data, f)
            
            return True
        except Exception as e:
            logger.error(f"Error storing vector to disk: {e}")
            return False
    
    def _get_vector_disk(self, path: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Fallback disk retrieval implementation"""
        try:
            filename = self._path_to_filename(path)
            filepath = os.path.join(self.cache_dir, filename)
            
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, "r") as f:
                data = json.load(f)
            
            vector = np.array(data["vector"], dtype=np.float32)
            metadata = data.get("metadata", {})
            
            return (vector, metadata)
        except Exception as e:
            logger.error(f"Error retrieving vector from disk: {e}")
            return None
    
    def _search_similar_vectors_disk(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Fallback disk search implementation"""
        try:
            results = []
            
            # Get all vector files
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith(".json"):
                    continue
                
                filepath = os.path.join(self.cache_dir, filename)
                
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    
                    vector = np.array(data["vector"], dtype=np.float32)
                    path = data["path"]
                    metadata = data.get("metadata", {})
                    
                    # Compute similarity
                    similarity = self._cosine_similarity(query_vector, vector)
                    
                    results.append((path, similarity, metadata))
                except Exception as e:
                    logger.error(f"Error processing vector file {filepath}: {e}")
            
            # Sort by similarity (highest first) and take top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error searching vectors on disk: {e}")
            return []
    
    def _delete_vector_disk(self, path: str) -> bool:
        """Fallback disk deletion implementation"""
        try:
            filename = self._path_to_filename(path)
            filepath = os.path.join(self.cache_dir, filename)
            
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error deleting vector from disk: {e}")
            return False
    
    def _get_all_paths_disk(self) -> List[str]:
        """Fallback disk path listing implementation"""
        try:
            paths = []
            
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith(".json"):
                    continue
                
                filepath = os.path.join(self.cache_dir, filename)
                
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    
                    path = data.get("path")
                    if path:
                        paths.append(path)
                except Exception as e:
                    logger.error(f"Error reading path from {filepath}: {e}")
            
            return paths
        except Exception as e:
            logger.error(f"Error getting paths from disk: {e}")
            return []
    
    def _clear_cache_disk(self) -> bool:
        """Fallback disk cache clearing implementation"""
        try:
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith(".json"):
                    continue
                
                filepath = os.path.join(self.cache_dir, filename)
                os.remove(filepath)
            
            return True
        except Exception as e:
            logger.error(f"Error clearing disk cache: {e}")
            return False
    
    def _get_stats_disk(self) -> Dict[str, Any]:
        """Fallback disk stats implementation"""
        try:
            # Count files and total size
            num_vectors = 0
            total_size = 0
            
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith(".json"):
                    continue
                
                filepath = os.path.join(self.cache_dir, filename)
                total_size += os.path.getsize(filepath)
                num_vectors += 1
            
            return {
                "num_vectors": num_vectors,
                "memory_usage": total_size,
                "memory_usage_human": f"{total_size / (1024*1024):.2f} MB",
                "redis_version": "N/A (using disk fallback)",
                "has_redisearch": False,
                "has_vector_search": False,
                "prefix": self.prefix,
                "vector_dim": self.vector_dim,
                "ttl": self.ttl,
                "cache_dir": self.cache_dir
            }
        except Exception as e:
            logger.error(f"Error getting disk stats: {e}")
            return {
                "error": str(e)
            }
    
    def _path_to_filename(self, path: str) -> str:
        """Convert a path to a safe filename"""
        # Replace unsafe characters
        safe_path = path.replace("/", "_").replace("\\", "_")
        return f"{safe_path}.json"
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        return dot_product / (norm_v1 * norm_v2)


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PathRAG Redis Integration")
    parser.add_argument("--host", default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument("--db", type=int, default=0, help="Redis database")
    parser.add_argument("--password", help="Redis password")
    parser.add_argument("--dim", type=int, default=384, help="Vector dimension")
    parser.add_argument("--prefix", default="pathrag", help="Key prefix")
    parser.add_argument("--ttl", type=int, default=604800, help="TTL in seconds (default: 7 days)")
    parser.add_argument("--cache-dir", help="Fallback disk cache directory")
    parser.add_argument("--test", action="store_true", help="Run tests")
    
    args = parser.parse_args()
    
    # Initialize adapter
    adapter = PathRAGRedisAdapter(
        host=args.host,
        port=args.port,
        db=args.db,
        password=args.password,
        vector_dim=args.dim,
        prefix=args.prefix,
        ttl=args.ttl,
        cache_dir=args.cache_dir
    )
    
    # Print stats
    stats = adapter.get_stats()
    print("\nPathRAG Redis Adapter Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Run tests if requested
    if args.test:
        print("\nRunning tests...")
        
        # Generate random vectors
        num_vectors = 100
        print(f"Generating {num_vectors} random vectors...")
        
        for i in range(num_vectors):
            vector = np.random.random(args.dim).astype(np.float32)
            path = f"test/path/{i}"
            metadata = {
                "id": i,
                "name": f"Test Vector {i}",
                "tags": ["test", "vector", f"id-{i}"]
            }
            
            success = adapter.store_vector(path, vector, metadata)
            if not success:
                print(f"Failed to store vector {i}")
        
        print("Vectors stored successfully")
        
        # Test retrieval
        print("\nTesting vector retrieval...")
        test_path = "test/path/42"
        result = adapter.get_vector(test_path)
        
        if result:
            vector, metadata = result
            print(f"Retrieved vector for {test_path}")
            print(f"  Vector shape: {vector.shape}")
            print(f"  Metadata: {metadata}")
        else:
            print(f"Failed to retrieve vector for {test_path}")
        
        # Test similarity search
        print("\nTesting similarity search...")
        query_vector = np.random.random(args.dim).astype(np.float32)
        results = adapter.search_similar_vectors(query_vector, top_k=5)
        
        print(f"Found {len(results)} similar vectors:")
        for path, score, metadata in results:
            print(f"  Path: {path}, Score: {score:.4f}, Metadata: {metadata}")
        
        # Test cache clearing
        print("\nClearing cache...")
        adapter.clear_cache()
        
        # Verify cache is empty
        paths = adapter.get_all_paths()
        print(f"Paths after clearing: {len(paths)}")
