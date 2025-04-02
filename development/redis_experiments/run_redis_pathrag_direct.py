#!/usr/bin/env python3
"""
Run PathRAG with Redis Integration (Direct Approach)

This script sets Redis environment variables and directly runs the PathRAG
embedding process from the rag-dataset-builder directory.

Usage:
    python run_redis_pathrag_direct.py [--gpu]
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run PathRAG with Redis integration (direct approach)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for embedding generation")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis database")
    parser.add_argument("--redis-password", default="", help="Redis password")
    parser.add_argument("--redis-prefix", default="pathrag", help="Redis key prefix")
    
    args = parser.parse_args()
    
    # Set environment variables for Redis
    os.environ["PATHRAG_REDIS_ENABLED"] = "true"
    os.environ["PATHRAG_REDIS_HOST"] = args.redis_host
    os.environ["PATHRAG_REDIS_PORT"] = str(args.redis_port)
    os.environ["PATHRAG_REDIS_DB"] = str(args.redis_db)
    os.environ["PATHRAG_REDIS_PREFIX"] = args.redis_prefix
    os.environ["PATHRAG_REDIS_TTL"] = str(86400 * 7)  # 7 days
    if args.redis_password:
        os.environ["PATHRAG_REDIS_PASSWORD"] = args.redis_password
    
    # Project paths
    project_root = os.path.abspath(os.path.dirname(__file__))
    rag_builder_dir = os.path.join(project_root, "rag-dataset-builder")
    
    # Build the command
    cmd = [
        sys.executable,
        "-m", "src.main",
        "--config", os.path.join(rag_builder_dir, "config", "config.yaml"),
        "--pathrag"
    ]
    
    if args.gpu:
        cmd.append("--gpu")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Execute the command
    try:
        # Change to the rag-dataset-builder directory
        os.chdir(rag_builder_dir)
        
        # Run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream the output
        for line in process.stdout:
            logger.info(line.strip())
        
        process.wait()
        if process.returncode != 0:
            logger.error(f"PathRAG embedding process failed with return code {process.returncode}")
            return process.returncode
        
        logger.info("PathRAG with Redis integration completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error running PathRAG with Redis integration: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
