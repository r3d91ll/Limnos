#!/usr/bin/env python3
"""
Redis-Enabled PathRAG Embedding Process

This script runs the PathRAG embedding process using Redis as a high-performance
caching solution instead of RAMDisk. It handles:
1. Loading source documents into Redis
2. Running the embedding process with GPU acceleration
3. Persisting the final database to disk

Usage:
    python redis_embedding.py --source-dir /path/to/source --output-dir /path/to/output [--gpu]
"""

import os
import sys
import time
import argparse
import logging
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, "logs", "redis_embedding.log"))
    ]
)
logger = logging.getLogger(__name__)

# Import Redis modules
try:
    from src.redis_cache import RedisCache
    from src.pathrag_redis_integration import PathRAGRedisAdapter
except ImportError:
    logger.error("Error: Redis modules not found. Make sure you're running from the correct directory.")
    sys.exit(1)

def load_documents_to_redis(source_dir, redis_cache):
    """Load source documents into Redis for faster processing"""
    logger.info(f"Loading documents from {source_dir} into Redis...")
    
    # Get list of files to process
    source_path = Path(source_dir)
    file_patterns = ["**/*.pdf", "**/*.txt", "**/*.md", "**/*.py", "**/*.js", "**/*.java"]
    exclude_patterns = ["**/README.md", "**/LICENSE.md", "**/.git/**", "**/node_modules/**"]
    
    # Find all files matching patterns
    all_files = []
    for pattern in file_patterns:
        all_files.extend(source_path.glob(pattern))
    
    # Filter out excluded files
    files_to_process = []
    for file_path in all_files:
        excluded = False
        for exclude in exclude_patterns:
            if file_path.match(exclude):
                excluded = True
                break
        if not excluded:
            files_to_process.append(file_path)
    
    logger.info(f"Found {len(files_to_process)} files to load into Redis")
    
    # Load files into Redis
    loaded_count = 0
    for file_path in files_to_process:
        try:
            # Read file content
            with open(file_path, "rb") as f:
                content = f.read()
            
            # Create a relative path from source_dir
            rel_path = str(file_path.relative_to(source_path))
            
            # Store in Redis with metadata
            metadata = {
                "path": rel_path,
                "size": len(content),
                "extension": file_path.suffix,
                "filename": file_path.name
            }
            
            # Use the file path as the key
            key = f"{redis_cache.prefix}:source:{rel_path}"
            
            # Store content in Redis
            redis_cache.redis_client.set(key, content)
            
            # Store metadata
            redis_cache.redis_client.hset(f"{key}:metadata", mapping=metadata)
            
            loaded_count += 1
            if loaded_count % 50 == 0:
                logger.info(f"Loaded {loaded_count}/{len(files_to_process)} files into Redis")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    logger.info(f"Successfully loaded {loaded_count} files into Redis")
    return loaded_count

def run_embedding_process(source_dir, output_dir, use_gpu=False):
    """Run the main embedding process with Redis integration"""
    logger.info(f"Running embedding process with Redis integration")
    logger.info(f"Source directory: {source_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"GPU mode: {use_gpu}")
    
    # Set environment variables for the embedding process
    env = os.environ.copy()
    env["PATHRAG_REDIS_ENABLED"] = "true"
    env["PATHRAG_REDIS_HOST"] = "localhost"
    env["PATHRAG_REDIS_PORT"] = "6379"
    env["PATHRAG_REDIS_DB"] = "0"
    env["PATHRAG_REDIS_PREFIX"] = "pathrag"
    env["PATHRAG_REDIS_TTL"] = str(86400 * 7)  # 7 days
    
    # Prepare the command
    cmd = [
        sys.executable,
        "-m", "rag-dataset-builder.src.main",  # Run as a module
        "--config", os.path.join(project_root, "rag-dataset-builder", "config", "config.yaml"),
        "--source", source_dir,
        "--output", output_dir,
        "--pathrag"
    ]
    
    if use_gpu:
        cmd.append("--gpu")
    
    # Run the embedding process
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream the output
        for line in process.stdout:
            logger.info(line.strip())
        
        process.wait()
        if process.returncode != 0:
            logger.error(f"Embedding process failed with return code {process.returncode}")
            return False
        
        logger.info("Embedding process completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error running embedding process: {e}")
        return False

def export_redis_to_disk(output_dir, redis_cache, adapter):
    """Export Redis data to disk for permanent storage"""
    logger.info(f"Exporting Redis data to disk at {output_dir}...")
    
    output_path = Path(output_dir)
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all keys with the prefix
    all_keys = redis_cache.redis_client.keys(f"{redis_cache.prefix}:*")
    logger.info(f"Found {len(all_keys)} keys in Redis with prefix '{redis_cache.prefix}'")
    
    # Export vector data
    vectors_dir = output_path / "vectors"
    vectors_dir.mkdir(exist_ok=True)
    
    # Get all vector keys
    vector_keys = [k.decode('utf-8') if isinstance(k, bytes) else k for k in all_keys 
                  if (k.decode('utf-8') if isinstance(k, bytes) else k).startswith(f"{redis_cache.prefix}:vector:")]
    
    exported_vectors = 0
    for key in vector_keys:
        try:
            # Extract path from key
            path = key.split(f"{redis_cache.prefix}:vector:")[1]
            
            # Get vector data
            vector_data = adapter.get_vector(path)
            if vector_data is None:
                continue
                
            vector, metadata = vector_data
            
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
                    import json
                    json.dump(metadata, f, indent=2)
            
            exported_vectors += 1
            if exported_vectors % 100 == 0:
                logger.info(f"Exported {exported_vectors}/{len(vector_keys)} vectors")
                
        except Exception as e:
            logger.error(f"Error exporting vector for key {key}: {e}")
    
    logger.info(f"Successfully exported {exported_vectors} vectors to disk")
    
    # Export metadata about the export
    stats = adapter.get_stats()
    with open(output_path / "redis_export_info.json", "w") as f:
        import json
        import datetime
        json.dump({
            "export_time": datetime.datetime.now().isoformat(),
            "redis_prefix": redis_cache.prefix,
            "num_vectors": stats.get("num_vectors", 0),
            "redis_version": stats.get("redis_version", ""),
            "vector_dim": stats.get("vector_dim", 0),
            "exported_vectors": exported_vectors
        }, f, indent=2)
    
    logger.info(f"Successfully exported Redis data to {output_dir}")
    return exported_vectors

def main():
    parser = argparse.ArgumentParser(description="Run PathRAG embedding process with Redis integration")
    parser.add_argument("--source-dir", required=True, help="Source documents directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for the database")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for embedding generation")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis database")
    parser.add_argument("--redis-password", default="", help="Redis password")
    parser.add_argument("--redis-prefix", default="pathrag", help="Redis key prefix")
    parser.add_argument("--clean", action="store_true", help="Clean existing data before starting")
    
    args = parser.parse_args()
    
    # Initialize Redis cache
    redis_cache = RedisCache(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        password=args.redis_password,
        prefix=args.redis_prefix
    )
    
    # Drop existing indices to avoid initialization errors
    try:
        # Get all indices
        indices = redis_cache.redis_client.execute_command("FT._LIST")
        for index in indices:
            index_name = index.decode('utf-8') if isinstance(index, bytes) else index
            if index_name.startswith(f"{args.redis_prefix}"):
                logger.info(f"Dropping existing index: {index_name}")
                try:
                    redis_cache.redis_client.execute_command(f"FT.DROPINDEX {index_name}")
                except Exception as e:
                    logger.warning(f"Error dropping index {index_name}: {e}")
    except Exception as e:
        logger.warning(f"Error listing indices: {e}")
    
    # Initialize PathRAG adapter
    adapter = PathRAGRedisAdapter(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        password=args.redis_password,
        prefix=args.redis_prefix
    )
    
    # Resolve paths
    source_dir = os.path.abspath(args.source_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Create directories if they don't exist
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean existing data if requested
    if args.clean:
        logger.info("Cleaning existing data...")
        if os.path.exists(output_dir):
            import shutil
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        
        # Clear Redis cache
        redis_cache.clear_cache()
    
    try:
        # Step 1: Load documents into Redis
        start_time = time.time()
        load_documents_to_redis(source_dir, redis_cache)
        load_time = time.time() - start_time
        logger.info(f"Document loading completed in {load_time:.2f} seconds")
        
        # Step 2: Run the embedding process
        start_time = time.time()
        success = run_embedding_process(source_dir, output_dir, args.gpu)
        embed_time = time.time() - start_time
        logger.info(f"Embedding process completed in {embed_time:.2f} seconds")
        
        if not success:
            logger.error("Embedding process failed")
            return 1
        
        # Step 3: Export Redis data to disk
        start_time = time.time()
        export_redis_to_disk(output_dir, redis_cache, adapter)
        export_time = time.time() - start_time
        logger.info(f"Data export completed in {export_time:.2f} seconds")
        
        logger.info("Redis-enabled PathRAG embedding process completed successfully")
        return 0
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error in Redis-enabled PathRAG embedding process: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    finally:
        # Ensure we clean up any resources
        logger.info("Cleaning up resources...")
        # Nothing to clean up for now, Redis will handle its own connections

if __name__ == "__main__":
    sys.exit(main())
