#!/usr/bin/env python3
"""
Redis Cache Module for PathRAG

This module provides a Redis-based caching solution for PathRAG, replacing the previous
RAMDisk implementation. It handles vector storage, retrieval, and management with
fallback mechanisms for different Redis versions.
"""

import os
import json
import time
import numpy as np
import redis
from typing import Dict, List, Tuple, Optional, Union, Any

class RedisCache:
    """Redis-based cache for PathRAG vectors and metadata"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        vector_dim: int = 384,
        prefix: str = "pathrag",
        ttl: int = 86400  # 24 hours default TTL
    ):
        """
        Initialize the Redis cache
        
        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            vector_dim: Dimension of vectors to store
            prefix: Prefix for all keys in Redis
            ttl: Time-to-live for cached items in seconds
        """
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False  # Keep binary data as is
        )
        self.vector_dim = vector_dim
        self.prefix = prefix
        self.ttl = ttl
        self.index_name = f"{prefix}_index"
        
        # Check connection
        try:
            self.redis_client.ping()
            print(f"Connected to Redis server at {host}:{port}")
        except redis.exceptions.ConnectionError as e:
            print(f"Error connecting to Redis: {e}")
            raise
        
        # Check if RediSearch module is loaded
        self.has_redisearch = False
        self.has_vector_search = False
        
        modules_info = self.redis_client.info('modules')
        if 'ft' in str(modules_info):
            self.has_redisearch = True
            print("RediSearch module is loaded")
            
            # Try to create a vector index to test if vector search is supported
            try:
                # First try to create a test index with vector search
                test_index = f"{self.prefix}_test_index"
                try:
                    self.redis_client.execute_command('FT.DROPINDEX', test_index)
                except:
                    pass
                
                self.redis_client.execute_command(
                    'FT.CREATE', test_index,
                    'ON', 'HASH',
                    'PREFIX', '1', f'{test_index}:',
                    'SCHEMA', 'vector', 'VECTOR', 'HNSW', '6', 'TYPE', 'FLOAT32', 
                    'DIM', vector_dim, 'DISTANCE_METRIC', 'COSINE'
                )
                
                # If we get here, vector search is supported
                self.has_vector_search = True
                print("Vector search is supported")
                
                # Clean up the test index
                self.redis_client.execute_command('FT.DROPINDEX', test_index)
            except redis.exceptions.ResponseError as e:
                if "Could not parse field spec" in str(e):
                    print("Vector search is not supported in this RediSearch version")
                else:
                    print(f"Error testing vector search: {e}")
        else:
            print("RediSearch module is not loaded - some functionality will be limited")
        
        # Initialize the main index
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize the main index for vector search or text fallback"""
        try:
            # Try to drop the index if it exists
            try:
                self.redis_client.execute_command('FT.DROPINDEX', self.index_name)
            except:
                pass
            
            if self.has_redisearch:
                if self.has_vector_search:
                    # Create a vector search index
                    self.redis_client.execute_command(
                        'FT.CREATE', self.index_name,
                        'ON', 'HASH',
                        'PREFIX', '1', f'{self.prefix}:',
                        'SCHEMA', 
                        'vector', 'VECTOR', 'HNSW', '6', 'TYPE', 'FLOAT32', 
                        'DIM', self.vector_dim, 'DISTANCE_METRIC', 'COSINE',
                        'metadata', 'TEXT', 'SORTABLE',
                        'path', 'TEXT', 'SORTABLE'
                    )
                    print(f"Created vector search index '{self.index_name}'")
                else:
                    # Create a text search index as fallback
                    self.redis_client.execute_command(
                        'FT.CREATE', self.index_name,
                        'ON', 'HASH',
                        'PREFIX', '1', f'{self.prefix}:',
                        'SCHEMA',
                        'metadata', 'TEXT', 'SORTABLE',
                        'path', 'TEXT', 'SORTABLE'
                    )
                    print(f"Created text search index '{self.index_name}' (vector search not supported)")
        except Exception as e:
            print(f"Error initializing index: {e}")
    
    def _get_key(self, path: str) -> str:
        """Get the Redis key for a path"""
        return f"{self.prefix}:{path}"
    
    def store_vector(
        self, 
        path: str, 
        vector: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a vector in Redis
        
        Args:
            path: Unique path identifier for the vector
            vector: The vector to store
            metadata: Optional metadata to store with the vector
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert vector to bytes
            vector_bytes = vector.astype(np.float32).tobytes()
            
            # Convert metadata to JSON string if provided
            metadata_str = json.dumps(metadata) if metadata else "{}"
            
            # Store in Redis
            key = self._get_key(path)
            mapping = {
                'vector': vector_bytes,
                'metadata': metadata_str,
                'path': path,
                'timestamp': time.time()
            }
            
            self.redis_client.hset(key, mapping=mapping)
            
            # Set TTL if specified
            if self.ttl > 0:
                self.redis_client.expire(key, self.ttl)
            
            return True
        except Exception as e:
            print(f"Error storing vector: {e}")
            return False
    
    def get_vector(self, path: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve a vector from Redis
        
        Args:
            path: Path identifier for the vector
            
        Returns:
            Tuple of (vector, metadata) or None if not found
        """
        try:
            key = self._get_key(path)
            data = self.redis_client.hgetall(key)
            
            if not data:
                return None
            
            # Extract vector
            vector_bytes = data.get(b'vector')
            if not vector_bytes:
                return None
                
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            
            # Extract metadata
            metadata_bytes = data.get(b'metadata', b'{}')
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            return (vector, metadata)
        except Exception as e:
            print(f"Error retrieving vector: {e}")
            return None
    
    def search_similar_vectors(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors in Redis
        
        Args:
            query_vector: The query vector
            top_k: Number of results to return
            
        Returns:
            List of tuples (path, score, metadata)
        """
        if not self.has_redisearch:
            # Fallback to basic key scan if RediSearch is not available
            return self._fallback_search(query_vector, top_k)
        
        if self.has_vector_search:
            # Use vector search if available
            return self._vector_search(query_vector, top_k)
        else:
            # Use text search as fallback
            return self._fallback_search(query_vector, top_k)
    
    def _vector_search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors using RediSearch vector search
        
        Args:
            query_vector: The query vector
            top_k: Number of results to return
            
        Returns:
            List of tuples (path, score, metadata)
        """
        try:
            # Convert query vector to bytes
            query_vector_bytes = query_vector.astype(np.float32).tobytes()
            
            # Perform the search
            results = self.redis_client.execute_command(
                'FT.SEARCH', self.index_name,
                f'*=>[KNN {top_k} @vector $query_vec AS score]',
                'PARAMS', '2', 'query_vec', query_vector_bytes,
                'RETURN', '3', 'score', 'path', 'metadata',
                'SORTBY', 'score',
                'DIALECT', '2'
            )
            
            # Parse results
            if results[0] == 0:
                return []
            
            parsed_results = []
            for i in range(1, len(results), 2):
                key = results[i].decode('utf-8')
                result_data = results[i+1]
                
                # Extract path, score and metadata
                path = None
                score = None
                metadata = {}
                
                for j in range(0, len(result_data), 2):
                    field_name = result_data[j].decode('utf-8')
                    if field_name == 'path':
                        path = result_data[j+1].decode('utf-8')
                    elif field_name == 'score':
                        score = float(result_data[j+1])
                    elif field_name == 'metadata':
                        try:
                            metadata = json.loads(result_data[j+1].decode('utf-8'))
                        except:
                            metadata = {}
                
                if path and score is not None:
                    parsed_results.append((path, score, metadata))
            
            return parsed_results
        except Exception as e:
            print(f"Vector search error: {e}")
            return self._fallback_search(query_vector, top_k)
    
    def _fallback_search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Fallback search method when vector search is not available
        
        This method retrieves all vectors and performs the similarity search in Python
        
        Args:
            query_vector: The query vector
            top_k: Number of results to return
            
        Returns:
            List of tuples (path, score, metadata)
        """
        try:
            # Get all keys with the prefix
            keys = self.redis_client.keys(f"{self.prefix}:*")
            
            if not keys:
                return []
            
            # Retrieve all vectors and compute similarities
            results = []
            for key in keys:
                data = self.redis_client.hgetall(key)
                
                if not data or b'vector' not in data:
                    continue
                
                # Extract vector
                vector_bytes = data[b'vector']
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
                
                # Extract path
                path = data.get(b'path', key).decode('utf-8')
                
                # Extract metadata
                metadata_bytes = data.get(b'metadata', b'{}')
                try:
                    metadata = json.loads(metadata_bytes.decode('utf-8'))
                except:
                    metadata = {}
                
                # Compute cosine similarity
                similarity = self._cosine_similarity(query_vector, vector)
                
                results.append((path, similarity, metadata))
            
            # Sort by similarity (highest first) and take top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"Fallback search error: {e}")
            return []
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        return dot_product / (norm_v1 * norm_v2)
    
    def delete_vector(self, path: str) -> bool:
        """
        Delete a vector from Redis
        
        Args:
            path: Path identifier for the vector
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            key = self._get_key(path)
            result = self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            print(f"Error deleting vector: {e}")
            return False
    
    def get_all_paths(self) -> List[str]:
        """
        Get all paths stored in Redis
        
        Returns:
            List of paths
        """
        try:
            keys = self.redis_client.keys(f"{self.prefix}:*")
            paths = []
            
            for key in keys:
                path = self.redis_client.hget(key, 'path')
                if path:
                    paths.append(path.decode('utf-8'))
                else:
                    # If path field is not available, use the key without prefix
                    key_str = key.decode('utf-8')
                    paths.append(key_str.replace(f"{self.prefix}:", "", 1))
            
            return paths
        except Exception as e:
            print(f"Error getting paths: {e}")
            return []
    
    def clear_cache(self) -> bool:
        """
        Clear all cached vectors
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            keys = self.redis_client.keys(f"{self.prefix}:*")
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dict with cache statistics
        """
        try:
            keys = self.redis_client.keys(f"{self.prefix}:*")
            num_vectors = len(keys)
            
            # Get memory usage
            memory_usage = 0
            for key in keys:
                key_memory = self.redis_client.memory_usage(key)
                if key_memory:
                    memory_usage += key_memory
            
            # Get Redis info
            redis_info = self.redis_client.info()
            
            return {
                'num_vectors': num_vectors,
                'memory_usage': memory_usage,
                'memory_usage_human': f"{memory_usage / (1024*1024):.2f} MB",
                'redis_version': redis_info.get('redis_version', 'unknown'),
                'has_redisearch': self.has_redisearch,
                'has_vector_search': self.has_vector_search,
                'prefix': self.prefix,
                'vector_dim': self.vector_dim,
                'ttl': self.ttl
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                'error': str(e)
            }


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Redis Cache for PathRAG")
    parser.add_argument("--host", default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument("--db", type=int, default=0, help="Redis database")
    parser.add_argument("--password", help="Redis password")
    parser.add_argument("--dim", type=int, default=384, help="Vector dimension")
    parser.add_argument("--prefix", default="pathrag", help="Key prefix")
    parser.add_argument("--ttl", type=int, default=86400, help="TTL in seconds")
    parser.add_argument("--test", action="store_true", help="Run tests")
    
    args = parser.parse_args()
    
    # Initialize cache
    cache = RedisCache(
        host=args.host,
        port=args.port,
        db=args.db,
        password=args.password,
        vector_dim=args.dim,
        prefix=args.prefix,
        ttl=args.ttl
    )
    
    # Print stats
    stats = cache.get_stats()
    print("\nRedis Cache Stats:")
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
            
            success = cache.store_vector(path, vector, metadata)
            if not success:
                print(f"Failed to store vector {i}")
        
        print("Vectors stored successfully")
        
        # Test retrieval
        print("\nTesting vector retrieval...")
        test_path = "test/path/42"
        result = cache.get_vector(test_path)
        
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
        results = cache.search_similar_vectors(query_vector, top_k=5)
        
        print(f"Found {len(results)} similar vectors:")
        for path, score, metadata in results:
            print(f"  Path: {path}, Score: {score:.4f}, Metadata: {metadata}")
        
        # Test cache clearing
        print("\nClearing cache...")
        cache.clear_cache()
        
        # Verify cache is empty
        paths = cache.get_all_paths()
        print(f"Paths after clearing: {len(paths)}")
