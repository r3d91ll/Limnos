#!/bin/bash
# Redis Integration for PathRAG
# This script modifies the run_unified.sh script to use Redis instead of RAMDisk
# for improved performance with vector storage and retrieval

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAIN_PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Redis is running
if ! redis-cli ping &> /dev/null; then
    print_error "Redis server is not running"
    print_status "Please start Redis with: sudo systemctl start redis-server"
    exit 1
fi

# Check if RediSearch module is loaded
if ! redis-cli module list | grep -q "ft"; then
    print_warning "RediSearch module is not loaded in Redis"
    print_status "Vector search functionality will be limited"
    print_status "You can install RediSearch with: sudo apt-get install redis-redisearch"
fi

# Check if the Redis cache module exists
if [ ! -f "${MAIN_PROJECT_ROOT}/src/redis_cache.py" ]; then
    print_error "Redis cache module not found at ${MAIN_PROJECT_ROOT}/src/redis_cache.py"
    exit 1
fi

# Check if the PathRAG Redis integration module exists
if [ ! -f "${MAIN_PROJECT_ROOT}/src/pathrag_redis_integration.py" ]; then
    print_error "PathRAG Redis integration module not found at ${MAIN_PROJECT_ROOT}/src/pathrag_redis_integration.py"
    exit 1
fi

# Create a backup of the original ramdisk_config.sh
RAMDISK_CONFIG="${PROJECT_ROOT}/scripts/ramdisk_config.sh"
RAMDISK_CONFIG_BACKUP="${RAMDISK_CONFIG}.bak.$(date +%Y%m%d%H%M%S)"

print_status "Creating backup of ramdisk_config.sh to ${RAMDISK_CONFIG_BACKUP}"
cp ${RAMDISK_CONFIG} ${RAMDISK_CONFIG_BACKUP}

# Create a new Redis configuration file
print_status "Creating Redis configuration for PathRAG..."
cat > ${PROJECT_ROOT}/scripts/redis_config.sh << 'EOF'
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
EOF

chmod +x ${PROJECT_ROOT}/scripts/redis_config.sh
print_status "Created Redis configuration at ${PROJECT_ROOT}/scripts/redis_config.sh"

# Create a backup of the original run_unified.sh
RUN_UNIFIED="${PROJECT_ROOT}/scripts/run_unified.sh"
RUN_UNIFIED_BACKUP="${RUN_UNIFIED}.bak.$(date +%Y%m%d%H%M%S)"

print_status "Creating backup of run_unified.sh to ${RUN_UNIFIED_BACKUP}"
cp ${RUN_UNIFIED} ${RUN_UNIFIED_BACKUP}

# Create a modified version of run_unified.sh that uses Redis
print_status "Creating Redis-enabled version of run_unified.sh..."
cat > ${PROJECT_ROOT}/scripts/run_unified_redis.sh << 'EOF'
#!/bin/bash
# Unified launcher for PathRAG Dataset Builder with Redis cache support
# Usage: ./scripts/run_unified_redis.sh [--gpu] [--pathrag|--graphrag|--literag] [--threads N] [--clean_db]
# Note: CPU mode is the default (no need to specify --cpu)

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAIN_PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
PYTHON_PATH=$(which python3)

# Default options
PROCESSING_MODE="cpu"  # CPU is the default processing mode
RAG_IMPL="pathrag"
CONFIG_DIR="config.d"
THREADS=24
CLEAN_DB=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      PROCESSING_MODE="gpu"
      shift
      ;;
    --pathrag)
      RAG_IMPL="pathrag"
      shift
      ;;
    --graphrag)
      RAG_IMPL="graphrag"
      shift
      ;;
    --literag)
      RAG_IMPL="literag"
      shift
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --clean_db)
      CLEAN_DB=true
      shift
      ;;
    --clean) # For backward compatibility
      echo "Warning: --clean is deprecated, use --clean_db instead"
      CLEAN_DB=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--cpu|--gpu] [--pathrag|--graphrag|--literag] [--threads N] [--clean_db]"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$PROCESSING_MODE" ]; then
    echo "Error: Processing mode (--cpu or --gpu) must be specified."
    echo "Usage: $0 [--cpu|--gpu] [--pathrag|--graphrag|--literag] [--threads N] [--clean]"
    exit 1
fi

# Configuration paths
MAIN_CONFIG="${PROJECT_ROOT}/config/config.yaml"
CONFIG_D_DIR="${PROJECT_ROOT}/${CONFIG_DIR}"

# Load Redis configuration
REDIS_CONFIG="${PROJECT_ROOT}/scripts/redis_config.sh"
echo "Loading Redis configuration from ${REDIS_CONFIG}..."

# Set default values
USE_REDIS=false

# Source the Redis configuration file
if [ -f "${REDIS_CONFIG}" ]; then
    source "${REDIS_CONFIG}"
else
    echo "Warning: Redis configuration file not found. Using defaults."
    # Default values
    REDIS_HOST="localhost"
    REDIS_PORT=6379
    REDIS_DB=0
    REDIS_PASSWORD=""
    REDIS_PREFIX="pathrag"
    VECTOR_DIM=384
    REDIS_TTL=604800
    ENABLE_REDIS=true
    USE_REDIS_CPU=true
    USE_REDIS_GPU=true
    SOURCE_DIR="../source_documents"
    OUTPUT_DIR="../rag_databases"
    PYTHON_PATH=$(which python3)
fi
    
# Resolve relative paths
if [[ "${OUTPUT_DIR}" == "../"* ]]; then
    DB_DIR="${PROJECT_ROOT}/../${OUTPUT_DIR#../}"
elif [[ "${OUTPUT_DIR}" == "./"* ]]; then
    DB_DIR="${PROJECT_ROOT}/${OUTPUT_DIR#./}"
elif [[ "${OUTPUT_DIR}" == /* ]]; then
    DB_DIR="${OUTPUT_DIR}"
else
    DB_DIR="${PROJECT_ROOT}/../${OUTPUT_DIR}"
fi

if [[ "${SOURCE_DIR}" == "../"* ]]; then
    SOURCE_DIR="${PROJECT_ROOT}/../${SOURCE_DIR#../}"
elif [[ "${SOURCE_DIR}" == "./"* ]]; then
    SOURCE_DIR="${PROJECT_ROOT}/${SOURCE_DIR#./}"
elif [[ "${SOURCE_DIR}" == /* ]]; then
    SOURCE_DIR="${SOURCE_DIR}"
else
    SOURCE_DIR="${PROJECT_ROOT}/../${SOURCE_DIR}"
fi

# Check if we should use a custom output directory from config.yaml or append RAG implementation
# If CUSTOM_OUTPUT_DIR is set to true, use DB_DIR as is without appending RAG_IMPL
CUSTOM_OUTPUT_DIR=true

# Export the CUSTOM_OUTPUT_DIR environment variable for the Python process
export CUSTOM_OUTPUT_DIR="$CUSTOM_OUTPUT_DIR"

if [ "$CUSTOM_OUTPUT_DIR" = "false" ]; then
    # Append RAG implementation to output directory if not already included
    if [[ ! "${DB_DIR}" == *"${RAG_IMPL}"* ]]; then
        DB_DIR="${DB_DIR}/${RAG_IMPL}"
    fi
fi

# Check if Redis should be used for the current processing mode
if [ "$ENABLE_REDIS" = "true" ]; then
    if [ "$PROCESSING_MODE" = "cpu" ] && [ "$USE_REDIS_CPU" = "true" ]; then
        USE_REDIS=true
    elif [ "$PROCESSING_MODE" = "gpu" ] && [ "$USE_REDIS_GPU" = "true" ]; then
        USE_REDIS=true
    fi
fi

echo "Using configuration:"
echo "Redis host: ${REDIS_HOST}:${REDIS_PORT}"
echo "Redis prefix: ${REDIS_PREFIX}"
echo "Source directory: ${SOURCE_DIR}"
echo "Output directory: ${DB_DIR}"
echo "RAG Implementation: ${RAG_IMPL}"
echo "Processing mode: ${PROCESSING_MODE}"
echo "Threads: ${THREADS}"
echo "Global Redis enabled: ${ENABLE_REDIS}"
echo "Using Redis for this run: ${USE_REDIS}"
echo "Clean database: ${CLEAN_DB}"

# Set up timestamp and logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/../logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/embedding_${RAG_IMPL}_${PROCESSING_MODE}_${TIMESTAMP}.log"
touch "${LOG_FILE}"

# Create directories if they don't exist
mkdir -p ${SOURCE_DIR}
mkdir -p ${DB_DIR}

# Clean existing database if requested
if [ "$CLEAN_DB" = true ]; then
    echo "Cleaning existing database in persistent storage..." | tee -a ${LOG_FILE}
    rm -rf ${DB_DIR}/*
    
    if [ "$USE_REDIS" = "true" ]; then
        echo "Clearing Redis cache..." | tee -a ${LOG_FILE}
        # Use the PathRAG Redis integration to clear the cache
        ${PYTHON_PATH} ${MAIN_PROJECT_ROOT}/src/pathrag_redis_integration.py --prefix ${REDIS_PREFIX} --clear-cache
    fi
fi

# Set up environment variables for Redis
if [ "$USE_REDIS" = "true" ]; then
    echo "Setting up Redis environment variables..." | tee -a ${LOG_FILE}
    export PATHRAG_REDIS_HOST="${REDIS_HOST}"
    export PATHRAG_REDIS_PORT="${REDIS_PORT}"
    export PATHRAG_REDIS_DB="${REDIS_DB}"
    export PATHRAG_REDIS_PASSWORD="${REDIS_PASSWORD}"
    export PATHRAG_REDIS_PREFIX="${REDIS_PREFIX}"
    export PATHRAG_REDIS_TTL="${REDIS_TTL}"
    export PATHRAG_REDIS_ENABLED="true"
    export PATHRAG_VECTOR_DIM="${VECTOR_DIM}"
else
    echo "Redis is disabled for ${PROCESSING_MODE} mode. Using disk storage directly." | tee -a ${LOG_FILE}
    export PATHRAG_REDIS_ENABLED="false"
fi

# Set up clean exit handler
function cleanup {
    echo "Cleaning up..." | tee -a ${LOG_FILE}
    
    # Ensure all Python processes are terminated
    echo "Ensuring all Python processes are terminated..." | tee -a ${LOG_FILE}
    # Find any Python processes related to our script and terminate them
    PYTHON_PIDS=$(ps aux | grep "[p]ython.*src.main" | awk '{print $2}')
    if [ -n "$PYTHON_PIDS" ]; then
        echo "Terminating Python processes: $PYTHON_PIDS" | tee -a ${LOG_FILE}
        for pid in $PYTHON_PIDS; do
            sudo kill -15 $pid 2>/dev/null || true
        done
        # Give processes time to terminate gracefully
        sleep 2
        # Force kill any remaining processes
        for pid in $PYTHON_PIDS; do
            sudo kill -9 $pid 2>/dev/null || true
        done
    fi
    
    echo "Cleanup complete" | tee -a ${LOG_FILE}
}

# Register cleanup handler for various exit signals
trap cleanup EXIT INT TERM

# Extract the subdirectory from config.yaml output_dir if it exists
CONFIG_OUTPUT_DIR=$(grep -E "^\s*output_dir:" ${MAIN_CONFIG} | awk '{print $2}' | tr -d '"'"'"')
if [ -n "${CONFIG_OUTPUT_DIR}" ]; then
    # If the config specifies a relative path, make it absolute
    if [[ "${CONFIG_OUTPUT_DIR}" != /* ]]; then
        CONFIG_OUTPUT_DIR="${PROJECT_ROOT}/${CONFIG_OUTPUT_DIR}"
    fi
    echo "Using output directory from config: ${CONFIG_OUTPUT_DIR}" | tee -a ${LOG_FILE}
    DB_DIR="${CONFIG_OUTPUT_DIR}"
fi

# Ensure the output directory exists
mkdir -p ${DB_DIR}

# Determine the Python executable to use
if [ -f "${PYTHON_PATH}" ]; then
    echo "Using Python: ${PYTHON_PATH}" | tee -a ${LOG_FILE}
else
    PYTHON_PATH=$(which python3)
    echo "Python path not found, using system Python: ${PYTHON_PATH}" | tee -a ${LOG_FILE}
fi

# Ensure Python environment is preserved when running with sudo
if [ -d "/home/todd/ML-Lab/New-HADES/.venv" ]; then
    export PYTHONPATH="/home/todd/ML-Lab/New-HADES:$PYTHONPATH"
    # If running as sudo, make sure to use the virtual environment
    if [ "$EUID" -eq 0 ]; then
        export PATH="/home/todd/ML-Lab/New-HADES/.venv/bin:$PATH"
    fi
fi

# Set environment variables for the Python process
export SOURCE_DIR="${SOURCE_DIR}"
export DB_DIR="${DB_DIR}"
export PROCESSING_MODE="${PROCESSING_MODE}"
export RAG_IMPL="${RAG_IMPL}"
export THREADS="${THREADS}"

# Run the main Python script
echo "Starting PathRAG Dataset Builder with Redis integration..." | tee -a ${LOG_FILE}
echo "Command: ${PYTHON_PATH} ${PROJECT_ROOT}/src/main.py --config ${MAIN_CONFIG} --config_d ${CONFIG_D_DIR} --${PROCESSING_MODE} --${RAG_IMPL} --threads ${THREADS}" | tee -a ${LOG_FILE}

${PYTHON_PATH} ${PROJECT_ROOT}/src/main.py --config ${MAIN_CONFIG} --config_d ${CONFIG_D_DIR} --${PROCESSING_MODE} --${RAG_IMPL} --threads ${THREADS} 2>&1 | tee -a ${LOG_FILE}

# Check the exit status
EXIT_STATUS=${PIPESTATUS[0]}
if [ ${EXIT_STATUS} -eq 0 ]; then
    echo "PathRAG Dataset Builder completed successfully" | tee -a ${LOG_FILE}
else
    echo "PathRAG Dataset Builder failed with exit code ${EXIT_STATUS}" | tee -a ${LOG_FILE}
fi

exit ${EXIT_STATUS}
EOF

chmod +x ${PROJECT_ROOT}/scripts/run_unified_redis.sh
print_status "Created Redis-enabled run script at ${PROJECT_ROOT}/scripts/run_unified_redis.sh"

# Create a Python module to integrate Redis with the PathRAG dataset builder
print_status "Creating PathRAG dataset builder Redis integration module..."
mkdir -p ${PROJECT_ROOT}/src/utils
cat > ${PROJECT_ROOT}/src/utils/redis_integration.py << 'EOF'
"""
Redis Integration for PathRAG Dataset Builder

This module provides integration between the PathRAG dataset builder and Redis,
allowing for high-performance vector storage and retrieval using Redis as a cache.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

# Add the main project root to the Python path
main_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if main_project_root not in sys.path:
    sys.path.insert(0, main_project_root)

# Import the PathRAG Redis adapter
try:
    from src.pathrag_redis_integration import PathRAGRedisAdapter
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class RedisIntegration:
    """Redis integration for PathRAG dataset builder"""
    
    def __init__(self):
        """Initialize the Redis integration"""
        self.redis_enabled = os.environ.get("PATHRAG_REDIS_ENABLED", "false").lower() == "true"
        
        if not self.redis_enabled:
            logger.info("Redis integration is disabled")
            return
        
        if not REDIS_AVAILABLE:
            logger.error("Redis integration is enabled but PathRAGRedisAdapter is not available")
            self.redis_enabled = False
            return
        
        # Get Redis configuration from environment variables
        self.redis_host = os.environ.get("PATHRAG_REDIS_HOST", "localhost")
        self.redis_port = int(os.environ.get("PATHRAG_REDIS_PORT", "6379"))
        self.redis_db = int(os.environ.get("PATHRAG_REDIS_DB", "0"))
        self.redis_password = os.environ.get("PATHRAG_REDIS_PASSWORD", "")
        self.redis_prefix = os.environ.get("PATHRAG_REDIS_PREFIX", "pathrag")
        self.redis_ttl = int(os.environ.get("PATHRAG_REDIS_TTL", "604800"))  # 7 days
        self.vector_dim = int(os.environ.get("PATHRAG_VECTOR_DIM", "384"))
        
        # Initialize the Redis adapter
        try:
            self.adapter = PathRAGRedisAdapter(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                vector_dim=self.vector_dim,
                prefix=self.redis_prefix,
                ttl=self.redis_ttl
            )
            logger.info(f"Redis integration initialized with prefix '{self.redis_prefix}'")
            
            # Log Redis stats
            stats = self.adapter.get_stats()
            logger.info(f"Redis stats: {stats}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis adapter: {e}")
            self.redis_enabled = False
    
    def store_vector(self, path: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a vector in Redis"""
        if not self.redis_enabled:
            return False
        
        try:
            return self.adapter.store_vector(path, vector, metadata)
        except Exception as e:
            logger.error(f"Error storing vector in Redis: {e}")
            return False
    
    def get_vector(self, path: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Retrieve a vector from Redis"""
        if not self.redis_enabled:
            return None
        
        try:
            return self.adapter.get_vector(path)
        except Exception as e:
            logger.error(f"Error retrieving vector from Redis: {e}")
            return None
    
    def search_similar_vectors(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors in Redis"""
        if not self.redis_enabled:
            return []
        
        try:
            return self.adapter.search_similar_vectors(query_vector, top_k)
        except Exception as e:
            logger.error(f"Error searching vectors in Redis: {e}")
            return []
    
    def delete_vector(self, path: str) -> bool:
        """Delete a vector from Redis"""
        if not self.redis_enabled:
            return False
        
        try:
            return self.adapter.delete_vector(path)
        except Exception as e:
            logger.error(f"Error deleting vector from Redis: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """Clear the Redis cache"""
        if not self.redis_enabled:
            return False
        
        try:
            return self.adapter.clear_cache()
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not self.redis_enabled:
            return {"redis_enabled": False}
        
        try:
            return self.adapter.get_stats()
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {"error": str(e)}


# Singleton instance
redis_integration = RedisIntegration()
EOF

print_status "Created PathRAG dataset builder Redis integration module at ${PROJECT_ROOT}/src/utils/redis_integration.py"

# Create a README file with instructions
print_status "Creating README file with instructions..."
cat > ${PROJECT_ROOT}/README_REDIS.md << 'EOF'
# Redis Integration for PathRAG

This document provides instructions for using Redis as a high-performance caching solution for PathRAG, replacing the previous RAMDisk implementation.

## Overview

Redis is used as an in-memory database to store vectors and metadata, providing similar performance benefits to RAMDisk while adding more features:

- **Vector Storage**: Efficiently store and retrieve embedding vectors
- **Vector Search**: Search for similar vectors using cosine similarity
- **Persistence**: Optional persistence to disk for durability
- **TTL Support**: Automatic expiration of cached items
- **Monitoring**: Built-in statistics and monitoring

## Requirements

- Redis server (version 6.0 or higher)
- RediSearch module (for vector search capabilities)
- Python 3.8 or higher
- Redis Python client (`pip install redis`)
- NumPy (`pip install numpy`)

## Setup

1. Install Redis and RediSearch:
   ```bash
   sudo apt-get update
   sudo apt-get install redis-server redis-tools redis-redisearch
   ```

2. Configure Redis for optimal performance:
   ```bash
   sudo cp /home/todd/ML-Lab/New-HADES/redis_test/redis-memory-optimized.conf /etc/redis/redis.conf
   sudo systemctl restart redis-server
   ```

3. Verify Redis is running with RediSearch:
   ```bash
   redis-cli ping
   redis-cli module list
   ```

## Usage

Instead of using the original `run_unified.sh` script, use the Redis-enabled version:

```bash
./scripts/run_unified_redis.sh [--gpu] [--pathrag|--graphrag|--literag] [--threads N] [--clean_db]
```

This script will automatically use Redis for caching vectors and metadata, providing similar performance benefits to RAMDisk.

## Configuration

Redis configuration is stored in `scripts/redis_config.sh`. You can modify this file to change Redis connection settings, TTL, and other parameters.

## Performance Considerations

- Redis is most effective when it has enough memory to store all vectors without swapping
- For a system with 256GB of RAM, allocate up to 200GB to Redis for optimal performance
- Monitor Redis memory usage with `redis-cli info memory`
- If Redis memory usage exceeds available RAM, consider reducing the TTL or clearing the cache more frequently

## Troubleshooting

If you encounter issues with Redis:

1. Check if Redis is running:
   ```bash
   systemctl status redis-server
   ```

2. Check Redis logs:
   ```bash
   journalctl -u redis-server
   ```

3. Verify RediSearch module is loaded:
   ```bash
   redis-cli module list
   ```

4. Test Redis connectivity:
   ```bash
   redis-cli ping
   ```

5. Check Redis memory usage:
   ```bash
   redis-cli info memory
   ```

## Reverting to RAMDisk

If you need to revert to the original RAMDisk implementation:

```bash
./scripts/run_unified.sh [--gpu] [--pathrag|--graphrag|--literag] [--threads N] [--clean_db]
```

This will use the original RAMDisk configuration without Redis.
EOF

print_status "Created README file at ${PROJECT_ROOT}/README_REDIS.md"

# Make the script executable
chmod +x ${PROJECT_ROOT}/scripts/redis_integration.sh

print_status "============================================="
print_status "Redis integration setup complete!"
print_status "To use Redis with PathRAG:"
print_status "1. Run the Redis-enabled script: ./scripts/run_unified_redis.sh"
print_status "2. See ${PROJECT_ROOT}/README_REDIS.md for detailed instructions"
print_status "============================================="
