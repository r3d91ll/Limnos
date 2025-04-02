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

# Pre-load source documents into Redis if Redis is enabled
if [ "$USE_REDIS" = "true" ]; then
    echo "Pre-loading source documents into Redis..." | tee -a ${LOG_FILE}
    
    # Create a Python script to load documents into Redis
    LOADER_SCRIPT="${PROJECT_ROOT}/scripts/load_documents_to_redis.py"
    
    cat > ${LOADER_SCRIPT} << 'EOF'
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
EOF
    
    chmod +x ${LOADER_SCRIPT}
    
    # Run the loader script
    ${PYTHON_PATH} ${LOADER_SCRIPT} \
        --source-dir ${SOURCE_DIR} \
        --redis-host ${REDIS_HOST} \
        --redis-port ${REDIS_PORT} \
        --redis-db ${REDIS_DB} \
        --redis-password "${REDIS_PASSWORD}" \
        --redis-prefix ${REDIS_PREFIX}
    
    echo "Source documents loaded into Redis" | tee -a ${LOG_FILE}
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
    
    # If Redis was used, ensure data is persisted to disk
    if [ "$USE_REDIS" = "true" ] && [ -n "${DB_DIR}" ] && [ -d "${DB_DIR}" ]; then
        echo "Persisting Redis data to disk at ${DB_DIR}..." | tee -a ${LOG_FILE}
        
        # Create a Python script to export Redis data to disk
        EXPORT_SCRIPT="${PROJECT_ROOT}/scripts/export_redis_to_disk.py"
        
        cat > ${EXPORT_SCRIPT} << 'EOF'
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

def export_to_disk(output_dir, redis_host, redis_port, redis_db, redis_password, redis_prefix):
    """Export Redis data to disk"""
    # Initialize Redis adapter
    adapter = PathRAGRedisAdapter(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password,
        prefix=redis_prefix
    )
    
    output_path = Path(output_dir)
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all keys with the prefix
    all_keys = adapter.redis_client.keys(f"{redis_prefix}:*")
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
                
                # Get document content
                content = adapter.redis_client.get(key)
                if content is None:
                    continue
                
                # Create directory structure
                path_obj = Path(path)
                parent_dir = source_dir / path_obj.parent
                parent_dir.mkdir(parents=True, exist_ok=True)
                
                # Save document to disk
                doc_file = source_dir / path
                with open(doc_file, "wb") as f:
                    f.write(content)
                    
                logger.debug(f"Exported source document {path}")
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
                data = adapter.redis_client.get(key)
                if data is None:
                    continue
                
                # Save index to disk
                index_file = index_dir / f"{name}.bin"
                with open(index_file, "wb") as f:
                    f.write(data)
                    
                logger.debug(f"Exported index {name}")
            except Exception as e:
                logger.error(f"Error exporting index for key {key}: {e}")
    
    # Export metadata about the export
    stats = adapter.get_stats()
    with open(output_path / "redis_export_info.json", "w") as f:
        json.dump({
            "export_time": datetime.datetime.now().isoformat(),
            "redis_prefix": redis_prefix,
            "num_vectors": stats.get("num_vectors", 0),
            "redis_version": stats.get("redis_version", ""),
            "vector_dim": stats.get("vector_dim", 0)
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
EOF
        
        chmod +x ${EXPORT_SCRIPT}
        
        # Run the export script
        ${PYTHON_PATH} ${EXPORT_SCRIPT} \
            --output-dir ${DB_DIR} \
            --redis-host ${REDIS_HOST} \
            --redis-port ${REDIS_PORT} \
            --redis-db ${REDIS_DB} \
            --redis-password "${REDIS_PASSWORD}" \
            --redis-prefix ${REDIS_PREFIX}
        
        echo "Redis data persisted to disk at ${DB_DIR}" | tee -a ${LOG_FILE}
    fi
    
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
CONFIG_OUTPUT_DIR=$(grep -E "^\s*output_dir:" ${MAIN_CONFIG} | awk '{print $2}' | tr -d '\"\'')
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
