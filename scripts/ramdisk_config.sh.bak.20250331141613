#!/bin/bash
# Shell-specific configuration for RAM disk
# This file is sourced by run_unified.sh

# RAM disk mount points (absolute paths, NO COMMENTS HERE)
SOURCE_RAMDISK_PATH="/tmp/ramdisk_source_documents"
DB_RAMDISK_PATH="/tmp/ramdisk_rag_databases"

# RAM disk sizes
SOURCE_RAMDISK_SIZE="20G"
DB_RAMDISK_SIZE="30G"

# Enable flags
ENABLE_RAMDISK=true
USE_RAMDISK_CPU=true
USE_RAMDISK_GPU=true

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
