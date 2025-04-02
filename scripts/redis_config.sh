#!/bin/bash
# Redis configuration for PathRAG
# This file replaces the RAMDisk configuration with Redis

# Redis connection settings
REDIS_HOST="localhost"
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=""

# Redis key prefix
REDIS_PREFIX="pathrag"

# Vector dimension
VECTOR_DIM=384

# TTL in seconds (7 days)
REDIS_TTL=604800

# Enable flags
ENABLE_REDIS=true
USE_REDIS_CPU=true
USE_REDIS_GPU=true

# Source and output directories (relative to project root)
SOURCE_DIR="../source_documents"
OUTPUT_DIR="../rag_databases"

# Python path (use virtual environment if available)
if [ -f "/home/todd/ML-Lab/New-HADES/.venv/bin/python" ]; then
    PYTHON_PATH="/home/todd/ML-Lab/New-HADES/.venv/bin/python"
    # Add the virtual environment to PATH when running as sudo
    export PATH="/home/todd/ML-Lab/New-HADES/.venv/bin:$PATH"
else
    PYTHON_PATH=$(which python3)
fi

# For backward compatibility with run_unified.sh
# These variables are not used with Redis but are kept to avoid errors
SOURCE_RAMDISK_PATH="/tmp/ramdisk_source_documents"
DB_RAMDISK_PATH="/tmp/ramdisk_rag_databases"
SOURCE_RAMDISK_SIZE="20G"
DB_RAMDISK_SIZE="30G"
ENABLE_RAMDISK=false
USE_RAMDISK_CPU=false
USE_RAMDISK_GPU=false
