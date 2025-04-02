#!/usr/bin/env python3
"""
Run PathRAG with Redis Integration

This script runs the PathRAG dataset builder with Redis integration for high-performance caching.
It sets up the necessary Redis environment variables and runs the existing rag-dataset-builder scripts.

Usage:
    python run_pathrag_redis.py --source-dir /path/to/source --output-dir /path/to/output [--gpu]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run PathRAG with Redis integration")
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
    
    # Resolve paths
    source_dir = os.path.abspath(args.source_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Create directories if they don't exist
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set environment variables for Redis
    os.environ["PATHRAG_REDIS_ENABLED"] = "true"
    os.environ["PATHRAG_REDIS_HOST"] = args.redis_host
    os.environ["PATHRAG_REDIS_PORT"] = str(args.redis_port)
    os.environ["PATHRAG_REDIS_DB"] = str(args.redis_db)
    os.environ["PATHRAG_REDIS_PREFIX"] = args.redis_prefix
    os.environ["PATHRAG_REDIS_TTL"] = str(86400 * 7)  # 7 days
    if args.redis_password:
        os.environ["PATHRAG_REDIS_PASSWORD"] = args.redis_password
    
    # Build the command
    project_root = os.path.abspath(os.path.dirname(__file__))
    rag_builder_dir = os.path.join(project_root, "rag-dataset-builder")
    
    # First, clear any existing Redis indices with the same prefix
    print(f"Clearing existing Redis indices with prefix '{args.redis_prefix}'...")
    redis_setup_script = os.path.join(project_root, "scripts", "setup_redis_cache.sh")
    if os.path.exists(redis_setup_script):
        subprocess.run([
            "bash", redis_setup_script, 
            "--drop-indices", 
            "--prefix", args.redis_prefix
        ], check=False)
    
    # Run the rag-dataset-builder script
    cmd = [
        "cd", rag_builder_dir, "&&",
        "python", "-m", "src.main",
        "--config", os.path.join(rag_builder_dir, "config", "config.yaml"),
        "--source", source_dir,
        "--output", output_dir,
        "--pathrag"
    ]
    
    if args.gpu:
        cmd.append("--gpu")
    
    # Convert to a shell command
    shell_cmd = " ".join(cmd)
    print(f"Running command: {shell_cmd}")
    
    # Execute the command
    process = subprocess.Popen(
        shell_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Stream the output
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    if process.returncode != 0:
        print(f"Error: Process exited with code {process.returncode}")
        return process.returncode
    
    print("PathRAG with Redis integration completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
