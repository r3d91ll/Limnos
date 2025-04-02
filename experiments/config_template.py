"""
Experiment Configuration Template

This file serves as a template for experiment configurations across all three phases.
Each phase should copy and modify this template as needed.
"""

from typing import Dict, Any, List, Optional

# Base configurations for all experiments
BASE_CONFIG = {
    # Database configurations
    "arango_host": "localhost",
    "arango_port": 8529,
    "arango_username": "root",
    "arango_password": "password",
    "arango_database": "pathrag",
    
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_username": "neo4j",
    "neo4j_password": "password",
    "neo4j_database": "graphrag",
    
    # Model configurations - will be overridden in phase-specific configs
    "model_name": "gpt2",  # Default for original implementations
    "openai_api_key": "",  # Required for original implementations
    
    # Embedding configurations
    "embedding_model": "multi-qa-MiniLM-L6-cos-v1",  # Default from PathRAG paper
    
    # Experiment parameters
    "random_seed": 42,
    "num_runs": 5,  # Number of runs per experiment for statistical significance
    
    # Dataset configurations
    "dataset_paths": {
        "kilt_nq": "data/datasets/kilt/nq-test.jsonl",
        "kilt_hotpotqa": "data/datasets/kilt/hotpotqa-test.jsonl",
    },
    
    # Metrics configurations
    "metrics": ["precision", "recall", "mrr", "ndcg", "rouge", "bleu", "bertscore"],
    
    # PathRAG specific configurations
    "pathrag_config": {
        "top_k": 5,
        "max_depth": 2,
        "max_paths": 3,
        "chunk_size": 300,
    },
    
    # GraphRAG specific configurations
    "graphrag_config": {
        "top_k": 5,
        "max_hops": 2,
        "community_detection": True,
    },
}

# Phase 1: Original Implementation configuration
PHASE1_CONFIG = {
    **BASE_CONFIG,
    "phase": "phase1",
    "description": "Original PathRAG and GraphRAG implementations",
    # Use original model configurations
}

# Phase 2: Qwen2.5 Coder Integration configuration
PHASE2_CONFIG = {
    **BASE_CONFIG,
    "phase": "phase2",
    "description": "PathRAG and GraphRAG with Qwen2.5 Coder",
    # Override model configurations for Qwen2.5
    "model_name": "qwen2.5-coder",
    "ollama_host": "localhost",
    "ollama_port": 11434,
}

# Phase 3: XnX Notation Integration configuration
PHASE3_CONFIG = {
    **BASE_CONFIG,
    "phase": "phase3",
    "description": "PathRAG and GraphRAG with XnX notation",
    # Original model variant
    "original_model": {
        "model_name": "gpt2",
        "openai_api_key": "",
    },
    # Qwen2.5 variant
    "qwen_model": {
        "model_name": "qwen2.5-coder",
        "ollama_host": "localhost",
        "ollama_port": 11434,
    },
    # XnX specific configurations
    "xnx_config": {
        "enable_weights": True,
        "enable_direction": True,
        "enable_temporal": False,  # Default to disabled for initial experiments
        "weight_decay_lambda": 0.1,  # For temporal weighting if enabled
    },
}
