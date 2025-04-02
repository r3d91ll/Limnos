#!/usr/bin/env python3
"""Load source documents into Redis for faster processing"""

import os
import sys
import glob
import hashlib
import argparse
import logging
from pathlib import Path

# Add the main project root to the Python path
main_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if main_project_root not in sys.path:
    sys.path.insert(0, main_project_root)

# Import the PathRAG Redis adapter
try:
    from src.pathrag_redis_integration import PathRAGRedisAdapter
except ImportError:
    print("Error: PathRAGRedisAdapter not found")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def load_documents(source_dir, redis_host, redis_port, redis_db, redis_password, redis_prefix):
    """Load documents from source_dir into Redis"""
    # Initialize Redis adapter
    adapter = PathRAGRedisAdapter(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password,
        prefix=redis_prefix
    )
    
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
    for file_path in files_to_process:
        try:
            # Read file content
            with open(file_path, "rb") as f:
                content = f.read()
            
            # Create a relative path from source_dir
            rel_path = file_path.relative_to(source_path)
            
            # Store in Redis with metadata
            metadata = {
                "path": str(rel_path),
                "size": len(content),
                "extension": file_path.suffix,
                "filename": file_path.name
            }
            
            # Use the file path as the key
            key = f"source:{rel_path}"
            
            # Store content in Redis
            adapter.redis_client.set(key, content)
            
            # Store metadata
            adapter.redis_client.hset(f"{key}:metadata", mapping=metadata)
            
            logger.debug(f"Loaded {rel_path} into Redis")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    logger.info(f"Successfully loaded {len(files_to_process)} files into Redis")

def main():
    parser = argparse.ArgumentParser(description="Load source documents into Redis")
    parser.add_argument("--source-dir", required=True, help="Source documents directory")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis database")
    parser.add_argument("--redis-password", default="", help="Redis password")
    parser.add_argument("--redis-prefix", default="pathrag", help="Redis key prefix")
    
    args = parser.parse_args()
    
    load_documents(
        args.source_dir,
        args.redis_host,
        args.redis_port,
        args.redis_db,
        args.redis_password,
        args.redis_prefix
    )

if __name__ == "__main__":
    main()
