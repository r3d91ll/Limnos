#!/usr/bin/env python3
"""
Preload Documents to Redis

This script preloads all source documents into Redis before starting the PathRAG embedding process.
This ensures that Redis is fully utilized for high-performance in-memory operations.
"""

import os
import sys
import argparse
import logging
import time
import redis
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def read_file_content(file_path):
    """Read the content of a file."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None

def load_document_to_redis(redis_client, file_path, redis_prefix):
    """Load a document into Redis."""
    try:
        # Read file content
        content = read_file_content(file_path)
        if content is None:
            return False
        
        # Create Redis keys
        relative_path = str(file_path)
        source_key = f"{redis_prefix}:source:{relative_path}"
        metadata_key = f"{source_key}:metadata"
        
        # Create metadata
        metadata = {
            "path": str(file_path),
            "size": len(content),
            "last_modified": os.path.getmtime(file_path),
            "created": os.path.getctime(file_path)
        }
        
        # Store in Redis
        redis_client.set(source_key, content)
        redis_client.set(metadata_key, json.dumps(metadata))
        
        return True
    except Exception as e:
        logger.error(f"Error loading document {file_path} to Redis: {e}")
        return False

def find_documents(source_dir, include_patterns=None, exclude_patterns=None):
    """Find all documents in the source directory that match the include/exclude patterns."""
    if include_patterns is None:
        include_patterns = ["**/*.pdf", "**/*.txt", "**/*.md", "**/*.py", "**/*.js", "**/*.java"]
    
    if exclude_patterns is None:
        exclude_patterns = ["**/README.md", "**/LICENSE.md", "**/.git/**", "**/node_modules/**"]
    
    documents = []
    for pattern in include_patterns:
        for file_path in Path(source_dir).glob(pattern):
            if file_path.is_file():
                # Check if file should be excluded
                exclude = False
                for exclude_pattern in exclude_patterns:
                    if file_path.match(exclude_pattern):
                        exclude = True
                        break
                
                if not exclude:
                    documents.append(file_path)
    
    return documents

def main():
    parser = argparse.ArgumentParser(description="Preload documents to Redis")
    parser.add_argument("--source-dir", required=True, help="Directory containing source documents")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis database")
    parser.add_argument("--redis-password", default="", help="Redis password")
    parser.add_argument("--redis-prefix", default="pathrag", help="Redis key prefix")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads for parallel loading")
    
    args = parser.parse_args()
    
    # Connect to Redis
    redis_client = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        password=args.redis_password if args.redis_password else None
    )
    
    try:
        # Test Redis connection
        redis_client.ping()
        logger.info(f"Successfully connected to Redis at {args.redis_host}:{args.redis_port}")
    except redis.ConnectionError:
        logger.error(f"Failed to connect to Redis at {args.redis_host}:{args.redis_port}")
        return 1
    
    # Find all documents
    logger.info(f"Finding documents in {args.source_dir}...")
    documents = find_documents(args.source_dir)
    logger.info(f"Found {len(documents)} documents")
    
    # Preload documents to Redis
    logger.info(f"Preloading {len(documents)} documents to Redis using {args.threads} threads...")
    start_time = time.time()
    
    success_count = 0
    failure_count = 0
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(load_document_to_redis, redis_client, doc, args.redis_prefix): doc for doc in documents}
        
        for i, future in enumerate(as_completed(futures), 1):
            doc = futures[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
                else:
                    failure_count += 1
                
                if i % 50 == 0 or i == len(documents):
                    logger.info(f"Loaded {i}/{len(documents)} documents into Redis")
            except Exception as e:
                logger.error(f"Error processing document {doc}: {e}")
                failure_count += 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"Document preloading completed in {duration:.2f} seconds")
    logger.info(f"Successfully loaded {success_count} documents")
    if failure_count > 0:
        logger.warning(f"Failed to load {failure_count} documents")
    
    # Get Redis memory usage
    info = redis_client.info(section="memory")
    memory_used = info.get("used_memory_human", "unknown")
    logger.info(f"Redis memory usage: {memory_used}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
