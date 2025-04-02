#!/usr/bin/env python3
"""Export Redis data to disk for permanent storage"""

import os
import sys
import json
import argparse
import logging
import datetime
import shutil
from pathlib import Path

# Add the main project root to the Python path
main_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if main_project_root not in sys.path:
    sys.path.insert(0, main_project_root)

# Import the PathRAG Redis adapter and Redis
try:
    from src.pathrag_redis_integration import PathRAGRedisAdapter
    import redis
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def export_to_disk(output_dir, redis_host, redis_port, redis_db, redis_password, redis_prefix):
    """Export Redis data to disk"""
    # Initialize Redis adapter for vector operations
    adapter = PathRAGRedisAdapter(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password,
        prefix=redis_prefix
    )
    
    # Create a direct Redis client for key operations
    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password,
        decode_responses=False
    )
    logger.info(f"Connected to Redis at {redis_host}:{redis_port} (DB: {redis_db})")
    
    output_path = Path(output_dir)
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all keys with the prefix
    all_keys = redis_client.keys(f"{redis_prefix}:*")
    logger.info(f"Found {len(all_keys)} keys in Redis with prefix '{redis_prefix}'")
    
    # Export vector data
    vectors_dir = output_path / "vectors"
    vectors_dir.mkdir(exist_ok=True)
    
    # Get all vector keys
    vector_keys = [k.decode('utf-8') if isinstance(k, bytes) else k for k in all_keys 
                  if (k.decode('utf-8') if isinstance(k, bytes) else k).startswith(f"{redis_prefix}:vector:")]
    
    for key in vector_keys:
        try:
            # Extract path from key
            path = key.split(f"{redis_prefix}:vector:")[1]
            
            # Get vector data
            try:
                # Try using the adapter first
                vector_data = adapter.get_vector(path)
                if vector_data is None:
                    # If adapter fails, try direct Redis access
                    vector_key = f"{redis_prefix}:vector:{path}"
                    vector_bytes = redis_client.get(vector_key)
                    if vector_bytes is None:
                        logger.warning(f"Vector not found for {path}")
                        continue
                    
                    # Convert bytes to numpy array
                    import numpy as np
                    vector = np.frombuffer(vector_bytes, dtype=np.float32)
                    
                    # Get metadata if available
                    metadata_key = f"{redis_prefix}:metadata:{path}"
                    metadata_bytes = redis_client.get(metadata_key)
                    metadata = json.loads(metadata_bytes.decode('utf-8')) if metadata_bytes else {}
                    
                    vector_data = (vector, metadata)
                
                vector, metadata = vector_data
            except Exception as e:
                logger.error(f"Error getting vector for {path}: {e}")
                continue
            
            # Create directory structure
            path_obj = Path(path)
            parent_dir = vectors_dir / path_obj.parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            
            # Save vector to disk
            vector_file = vectors_dir / f"{path}.npy"
            with open(vector_file, "wb") as f:
                import numpy as np
                np.save(f, vector)
            
            # Save metadata
            if metadata:
                metadata_file = vectors_dir / f"{path}.json"
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)
                    
            logger.debug(f"Exported vector for {path}")
        except Exception as e:
            logger.error(f"Error exporting vector for key {key}: {e}")
    
    # Export source documents
    source_keys = [k.decode('utf-8') if isinstance(k, bytes) else k for k in all_keys 
                  if (k.decode('utf-8') if isinstance(k, bytes) else k).startswith(f"{redis_prefix}:source:") 
                  and not (k.decode('utf-8') if isinstance(k, bytes) else k).endswith(":metadata")]
    
    if source_keys:
        source_dir = output_path / "source_documents"
        source_dir.mkdir(exist_ok=True)
        
        for key in source_keys:
            try:
                # Extract path from key
                path = key.split(f"{redis_prefix}:source:")[1]
                
                # Get document content from Redis
                try:
                    content = redis_client.get(key)
                    if content is None:
                        logger.warning(f"Document content not found for {path}")
                        continue
                except Exception as e:
                    logger.error(f"Error getting document content for {path}: {e}")
                    continue
                
                # Create a safe filename by replacing problematic characters
                safe_filename = path.split('/')[-1]
                if not safe_filename:
                    safe_filename = "unknown_document.txt"
                
                # Create directory structure based on document type
                doc_type = "other"
                if "pdf" in path.lower():
                    doc_type = "pdf"
                elif ".txt" in path.lower():
                    doc_type = "text"
                elif ".md" in path.lower():
                    doc_type = "markdown"
                elif ".html" in path.lower() or ".htm" in path.lower():
                    doc_type = "html"
                
                type_dir = source_dir / doc_type
                type_dir.mkdir(parents=True, exist_ok=True)
                
                # Save document to disk with a safe path
                doc_file = type_dir / safe_filename
                try:
                    with open(doc_file, "wb") as f:
                        f.write(content)
                    logger.debug(f"Exported source document {path} to {doc_file}")
                except Exception as e:
                    logger.error(f"Error writing document to {doc_file}: {e}")
            except Exception as e:
                logger.error(f"Error exporting source document for key {key}: {e}")
    
    # Export index data if it exists
    index_keys = [k.decode('utf-8') if isinstance(k, bytes) else k for k in all_keys 
                 if "index" in (k.decode('utf-8') if isinstance(k, bytes) else k).lower() 
                 and not (k.decode('utf-8') if isinstance(k, bytes) else k).endswith(":metadata")]
    
    if index_keys:
        index_dir = output_path / "indexes"
        index_dir.mkdir(exist_ok=True)
        
        for key in index_keys:
            try:
                # Extract name from key
                name = key.split(f"{redis_prefix}:")[1]
                
                # Get index data
                try:
                    data = redis_client.get(key)
                    if data is None:
                        logger.warning(f"Index data not found for {name}")
                        continue
                except Exception as e:
                    logger.error(f"Error getting index data for {name}: {e}")
                    continue
                
                # Create a safe filename by hashing the original name if it's too long
                safe_name = name
                if len(name) > 100 or '/' in name:
                    import hashlib
                    safe_name = hashlib.md5(name.encode()).hexdigest()
                    logger.debug(f"Converted long index name to hash: {name} -> {safe_name}")
                
                # Save index to disk
                index_file = index_dir / f"{safe_name}.bin"
                try:
                    with open(index_file, "wb") as f:
                        f.write(data)
                    logger.debug(f"Exported index {name} to {index_file}")
                except Exception as e:
                    logger.error(f"Error writing index to {index_file}: {e}")
            except Exception as e:
                logger.error(f"Error exporting index for key {key}: {e}")
    
    # Export metadata about the export
    try:
        # Get Redis info for stats
        redis_info = redis_client.info()
        redis_version = redis_info.get('redis_version', '')
        
        # Count vectors
        vector_keys_count = len([k for k in all_keys if (k.decode('utf-8') if isinstance(k, bytes) else k).startswith(f"{redis_prefix}:vector:")])
        
        # Try to determine vector dimension from a sample vector
        vector_dim = 0
        if vector_keys and len(vector_keys) > 0:
            try:
                sample_key = vector_keys[0]
                sample_path = sample_key.split(f"{redis_prefix}:vector:")[1]
                sample_vector_bytes = redis_client.get(sample_key)
                if sample_vector_bytes:
                    import numpy as np
                    sample_vector = np.frombuffer(sample_vector_bytes, dtype=np.float32)
                    vector_dim = len(sample_vector)
            except Exception as e:
                logger.warning(f"Could not determine vector dimension: {e}")
        
        # Write stats to file
        with open(output_path / "redis_export_info.json", "w") as f:
            json.dump({
                "export_time": datetime.datetime.now().isoformat(),
                "redis_prefix": redis_prefix,
                "num_vectors": vector_keys_count,
                "redis_version": redis_version,
                "vector_dim": vector_dim,
                "total_keys": len(all_keys),
                "memory_used": redis_info.get('used_memory_human', 'unknown')
            }, f, indent=2)
    except Exception as e:
        logger.error(f"Error exporting metadata: {e}")
        # Create a minimal metadata file
        with open(output_path / "redis_export_info.json", "w") as f:
            json.dump({
                "export_time": datetime.datetime.now().isoformat(),
                "redis_prefix": redis_prefix,
                "error": str(e)
            }, f, indent=2)
    
    logger.info(f"Successfully exported Redis data to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Export Redis data to disk")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis database")
    parser.add_argument("--redis-password", default="", help="Redis password")
    parser.add_argument("--redis-prefix", default="pathrag", help="Redis key prefix")
    
    args = parser.parse_args()
    
    export_to_disk(
        args.output_dir,
        args.redis_host,
        args.redis_port,
        args.redis_db,
        args.redis_password,
        args.redis_prefix
    )

if __name__ == "__main__":
    main()
