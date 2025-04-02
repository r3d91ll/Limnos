# Redis Integration Experiments

This directory contains experimental scripts that were created during the development of the Redis integration for the PathRAG system. These scripts are kept for reference purposes but have been superseded by the unified `hades_unified.py` script in the project root.

## Scripts

- `redis_embedding.py`: Initial script for Redis-enabled PathRAG embedding process
- `redis_pathrag_simple.py`: Simplified script for Redis integration with PathRAG
- `run_pathrag_with_redis.py`: Script to run PathRAG with Redis integration
- `run_pathrag_redis.py`: Another approach to running PathRAG with Redis
- `run_redis_pathrag_complete.sh`: Shell script for the complete Redis-enabled PathRAG workflow
- `run_redis_pathrag_direct.py`: Direct approach to running PathRAG with Redis
- `run_redis_pathrag_env.sh`: Shell script to set Redis environment variables and run PathRAG
- `run_redis_pathrag.sh`: Original shell script for Redis-enabled PathRAG
- `preload_documents_to_redis.py`: Script to preload documents into Redis before embedding

## Usage

These scripts are kept for reference only and should not be used directly. Instead, use the unified `hades_unified.py` script in the project root, which provides a comprehensive interface for running the PathRAG system with Redis integration.

```bash
# Activate the virtual environment
source .venv/bin/activate

# Create a new dataset with GPU acceleration
python hades_unified.py create --source-dir /path/to/source --output-dir /path/to/output --gpu

# For more information
python hades_unified.py --help
```

## Development History

These scripts were created during the development of the Redis integration for the PathRAG system, which replaced the previous RAMDisk approach with a more flexible and powerful Redis-based solution. The development process involved several iterations to optimize the Redis integration and ensure proper handling of document loading, embedding, and database export.

The final solution, implemented in `hades_unified.py`, provides a unified interface for all PathRAG operations, with clear separation between different modes (create, infer, retrieve) and comprehensive Redis integration.
