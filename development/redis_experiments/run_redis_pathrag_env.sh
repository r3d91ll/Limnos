#!/bin/bash
# Script to run PathRAG with Redis integration
# This script ensures all Redis environment variables are properly set

# Set Redis environment variables
export PATHRAG_REDIS_ENABLED=true
export PATHRAG_REDIS_HOST=localhost
export PATHRAG_REDIS_PORT=6379
export PATHRAG_REDIS_DB=0
export PATHRAG_REDIS_PREFIX=pathrag
export PATHRAG_REDIS_TTL=604800  # 7 days in seconds

# Print Redis configuration
echo "Redis configuration:"
echo "PATHRAG_REDIS_ENABLED: $PATHRAG_REDIS_ENABLED"
echo "PATHRAG_REDIS_HOST: $PATHRAG_REDIS_HOST"
echo "PATHRAG_REDIS_PORT: $PATHRAG_REDIS_PORT"
echo "PATHRAG_REDIS_DB: $PATHRAG_REDIS_DB"
echo "PATHRAG_REDIS_PREFIX: $PATHRAG_REDIS_PREFIX"
echo "PATHRAG_REDIS_TTL: $PATHRAG_REDIS_TTL"

# Set project paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_BUILDER_DIR="$PROJECT_ROOT/rag-dataset-builder"
SOURCE_DIR="$PROJECT_ROOT/source_documents"
OUTPUT_DIR="$PROJECT_ROOT/rag_databases/pathrag_redis_gpu_test1"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Use GPU if specified
GPU_FLAG=""
if [ "$1" == "--gpu" ]; then
  GPU_FLAG="--gpu"
  echo "Using GPU for embedding generation"
fi

# Change to the rag-dataset-builder directory
cd "$RAG_BUILDER_DIR" || { echo "Failed to change to $RAG_BUILDER_DIR"; exit 1; }

# Run the PathRAG embedding process
echo "Running PathRAG with Redis integration..."
python -m src.main --config "$RAG_BUILDER_DIR/config/config.yaml" --pathrag $GPU_FLAG
