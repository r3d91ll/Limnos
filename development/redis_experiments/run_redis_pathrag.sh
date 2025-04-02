#!/bin/bash
# Run PathRAG with Redis integration
# This script sets up Redis environment variables and runs the PathRAG embedding process

set -e

# Default values
SOURCE_DIR="/home/todd/ML-Lab/New-HADES/source_documents"
OUTPUT_DIR="/home/todd/ML-Lab/New-HADES/rag_databases/pathrag_redis_gpu_test1"
USE_GPU=false
REDIS_HOST="localhost"
REDIS_PORT=6379
REDIS_DB=0
REDIS_PREFIX="pathrag"
REDIS_PASSWORD=""
REDIS_TTL=604800  # 7 days in seconds

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --gpu)
      USE_GPU=true
      shift
      ;;
    --redis-host)
      REDIS_HOST="$2"
      shift 2
      ;;
    --redis-port)
      REDIS_PORT="$2"
      shift 2
      ;;
    --redis-db)
      REDIS_DB="$2"
      shift 2
      ;;
    --redis-prefix)
      REDIS_PREFIX="$2"
      shift 2
      ;;
    --redis-password)
      REDIS_PASSWORD="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create directories if they don't exist
mkdir -p "$SOURCE_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Set environment variables for Redis
export PATHRAG_REDIS_ENABLED=true
export PATHRAG_REDIS_HOST="$REDIS_HOST"
export PATHRAG_REDIS_PORT="$REDIS_PORT"
export PATHRAG_REDIS_DB="$REDIS_DB"
export PATHRAG_REDIS_PREFIX="$REDIS_PREFIX"
export PATHRAG_REDIS_TTL="$REDIS_TTL"

if [ -n "$REDIS_PASSWORD" ]; then
  export PATHRAG_REDIS_PASSWORD="$REDIS_PASSWORD"
fi

# Print configuration
echo "Running PathRAG with Redis integration"
echo "Source directory: $SOURCE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "GPU mode: $USE_GPU"
echo "Redis host: $REDIS_HOST"
echo "Redis port: $REDIS_PORT"
echo "Redis database: $REDIS_DB"
echo "Redis prefix: $REDIS_PREFIX"

# Clear existing Redis indices with the same prefix
echo "Clearing existing Redis indices with prefix '$REDIS_PREFIX'..."
redis-cli FT._LIST 2>/dev/null | grep "$REDIS_PREFIX" | while read -r index; do
  echo "Dropping index: $index"
  redis-cli FT.DROPINDEX "$index" 2>/dev/null || true
done

# Load documents into Redis
echo "Loading documents into Redis..."
python3 <<EOF
import os
import sys
import time
from pathlib import Path
import redis

# Initialize Redis connection
redis_client = redis.Redis(
    host="$REDIS_HOST",
    port=$REDIS_PORT,
    db=$REDIS_DB,
    password="$REDIS_PASSWORD" if "$REDIS_PASSWORD" else None,
    decode_responses=False
)

# Check Redis connection
try:
    redis_client.ping()
    print("Connected to Redis server")
except Exception as e:
    print(f"Error connecting to Redis: {e}")
    sys.exit(1)

# Get list of files to process
source_path = Path("$SOURCE_DIR")
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

print(f"Found {len(files_to_process)} files to load into Redis")

# Load files into Redis
loaded_count = 0
for file_path in files_to_process:
    try:
        # Read file content
        with open(file_path, "rb") as f:
            content = f.read()
        
        # Create a relative path from source_dir
        rel_path = str(file_path.relative_to(source_path))
        
        # Store in Redis with metadata
        metadata = {
            "path": rel_path,
            "size": len(content),
            "extension": file_path.suffix,
            "filename": file_path.name
        }
        
        # Use the file path as the key
        key = f"$REDIS_PREFIX:source:{rel_path}"
        
        # Store content in Redis
        redis_client.set(key, content)
        
        # Store metadata
        redis_client.hset(f"{key}:metadata", mapping=metadata)
        
        loaded_count += 1
        if loaded_count % 50 == 0:
            print(f"Loaded {loaded_count}/{len(files_to_process)} files into Redis")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

print(f"Successfully loaded {loaded_count} files into Redis")
EOF

# Run the PathRAG embedding process
echo "Running PathRAG embedding process..."
cd /home/todd/ML-Lab/New-HADES/rag-dataset-builder

# Create a custom config file with the proper log directory
CONFIG_FILE="$OUTPUT_DIR/custom_config.yaml"
cp /home/todd/ML-Lab/New-HADES/rag-dataset-builder/config/config.yaml "$CONFIG_FILE"

# Update the logs directory in the config file
sed -i "s|logs: \"../logs\"|logs: \"$OUTPUT_DIR/logs\"|g" "$CONFIG_FILE"

# Build the command
CMD="python -m src.main --config $CONFIG_FILE --source $SOURCE_DIR --output $OUTPUT_DIR --pathrag"

if [ "$USE_GPU" = true ]; then
  CMD="$CMD --gpu"
fi

echo "Running command: $CMD"
eval "$CMD"

# Export Redis data to disk
echo "Exporting Redis data to disk..."
python3 <<EOF
import os
import sys
import time
import json
import datetime
from pathlib import Path
import redis
import numpy as np

# Initialize Redis connection
redis_client = redis.Redis(
    host="$REDIS_HOST",
    port=$REDIS_PORT,
    db=$REDIS_DB,
    password="$REDIS_PASSWORD" if "$REDIS_PASSWORD" else None,
    decode_responses=False
)

# Check Redis connection
try:
    redis_client.ping()
    print("Connected to Redis server")
except Exception as e:
    print(f"Error connecting to Redis: {e}")
    sys.exit(1)

output_dir = "$OUTPUT_DIR"
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

# Get all keys with the prefix
all_keys = redis_client.keys(f"$REDIS_PREFIX:*")
print(f"Found {len(all_keys)} keys in Redis with prefix '$REDIS_PREFIX'")

# Export vector data
vectors_dir = output_path / "vectors"
vectors_dir.mkdir(exist_ok=True)

# Get all vector keys
vector_keys = [k.decode('utf-8') if isinstance(k, bytes) else k for k in all_keys 
              if (k.decode('utf-8') if isinstance(k, bytes) else k).startswith(f"$REDIS_PREFIX:vector:")]

exported_vectors = 0
for key in vector_keys:
    try:
        # Extract path from key
        path = key.split(f"$REDIS_PREFIX:vector:")[1]
        
        # Get vector data
        vector_data = redis_client.get(key)
        if vector_data is None:
            continue
            
        # Get metadata
        metadata_key = f"{key}:metadata"
        metadata_data = redis_client.hgetall(metadata_key)
        metadata = {k.decode('utf-8') if isinstance(k, bytes) else k: 
                   v.decode('utf-8') if isinstance(v, bytes) else v 
                   for k, v in metadata_data.items()} if metadata_data else {}
        
        # Convert vector data to numpy array
        vector = np.frombuffer(vector_data, dtype=np.float32)
        
        # Create directory structure
        path_obj = Path(path)
        parent_dir = vectors_dir / path_obj.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        
        # Save vector to disk
        vector_file = vectors_dir / f"{path}.npy"
        with open(vector_file, "wb") as f:
            np.save(f, vector)
        
        # Save metadata
        if metadata:
            metadata_file = vectors_dir / f"{path}.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        
        exported_vectors += 1
        if exported_vectors % 100 == 0:
            print(f"Exported {exported_vectors}/{len(vector_keys)} vectors")
            
    except Exception as e:
        print(f"Error exporting vector for key {key}: {e}")

print(f"Successfully exported {exported_vectors} vectors to disk")

# Export metadata about the export
with open(output_path / "redis_export_info.json", "w") as f:
    json.dump({
        "export_time": datetime.datetime.now().isoformat(),
        "redis_prefix": "$REDIS_PREFIX",
        "num_vectors": exported_vectors,
        "redis_version": redis_client.info().get("redis_version", ""),
        "exported_vectors": exported_vectors
    }, f, indent=2)

print(f"Successfully exported Redis data to {output_dir}")
EOF

echo "PathRAG with Redis integration completed successfully"
