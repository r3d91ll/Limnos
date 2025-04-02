#!/bin/bash
# Simple launcher for PathRAG Dataset Builder
# Usage: ./scripts/run.sh [--cpu|--gpu] [--pathrag|--graphrag|--literag]

cd "$(dirname "$0")/.."
PYTHON_PATH=$(which python3)

# Default options
PROCESSING_MODE=""
RAG_IMPL="--pathrag"
CONFIG_DIR="config.d"
THREADS=24

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --cpu)
      PROCESSING_MODE="--cpu"
      shift
      ;;
    --gpu)
      PROCESSING_MODE="--gpu"
      shift
      ;;
    --pathrag|--graphrag|--literag)
      RAG_IMPL="$1"
      shift
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--cpu|--gpu] [--pathrag|--graphrag|--literag] [--threads N]"
      exit 1
      ;;
  esac
done

# Run the dataset builder
echo "Running PathRAG Dataset Builder with options: ${PROCESSING_MODE} ${RAG_IMPL} --threads ${THREADS}"
${PYTHON_PATH} -m src.main --config_dir ${CONFIG_DIR} ${PROCESSING_MODE} ${RAG_IMPL} --threads ${THREADS}
