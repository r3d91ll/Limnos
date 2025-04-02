"""
Configuration module for HADES XnX experiments.

This module provides centralized configuration for all experimental phases,
ensuring consistent settings and reproducible results.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "datasets"
RESULTS_DIR = DATA_DIR / "results"

# Database configurations
ARANGO_CONFIG = {
    "host": "localhost",
    "port": 8529,
    "username": "root",
    "password": "",  # Set via environment variable in production
    "database": "pathrag"
}

NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "",  # Set via environment variable in production
    "database": "graphrag"
}

# Model configurations
EMBEDDING_MODELS = {
    "default": "multi-qa-MiniLM-L6-cos-v1",  # Default from PathRAG paper
    "alternative": "all-mpnet-base-v2"  # Higher quality alternative
}

# Ollama configuration for Qwen2.5
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "model": "qwen2.5-coder",
    "timeout": 120,
}

# Original model configurations (from papers)
ORIGINAL_MODEL_CONFIGS = {
    "pathrag": {
        "model": "gpt-3.5-turbo",  # Update with actual model from paper
        "temperature": 0.0,
        "max_tokens": 512,
        "top_p": 1.0,
    },
    "graphrag": {
        "model": "gpt-3.5-turbo",  # Update with actual model from paper
        "temperature": 0.0,
        "max_tokens": 512,
        "top_p": 1.0,
    }
}

# Experiment configurations
DEFAULT_EXPERIMENT_CONFIG = {
    "num_runs": 5,                # Number of runs for each experiment
    "random_seed": 42,            # Base random seed
    "top_k_retrieval": [1, 3, 5], # k values for precision/recall@k
    "max_path_depth": 2,          # Default path depth for traversal
    "max_paths": 3,               # Default number of paths to include
    "chunk_size": 300,            # Default chunk size for text
}

# Dataset configurations
DATASETS = {
    "hotpotqa": {
        "path": str(DATASET_DIR / "hotpotqa"),
        "train_file": "train.json",
        "dev_file": "dev.json", 
        "test_file": "test.json",
    },
    "nq": {
        "path": str(DATASET_DIR / "natural_questions"),
        "train_file": "train.json",
        "dev_file": "dev.json",
        "test_file": "test.json",
    },
    # Add other datasets as needed
}

# XnX notation configurations for Phase 3
XNX_CONFIGS = {
    "default": {
        "path_formatter": "standard",
        "weight_decay": 0.9,      # Weight decay factor for hop distance
        "relation_weights": {
            "next_chunk": 0.95,   # Sequential relationship
            "entity_link": 0.85,  # Entity-based relationship
            "contains": 0.99,     # Hierarchical relationship
            "imports": 0.9,       # Import relationship
            "default": 0.8,       # Default for other relationships
        },
    },
    "advanced": {
        "path_formatter": "detailed",
        "weight_decay": 0.85,
        "relation_weights": {
            "next_chunk": 0.92,
            "entity_link": 0.88,
            "contains": 0.98,
            "imports": 0.85,
            "default": 0.75,
        },
    }
}

# Evaluation metrics configuration
METRICS_CONFIG = {
    "retrieval": ["precision", "recall", "mrr", "ndcg"],
    "generation": ["rouge", "bleu", "bertscore"],
    "efficiency": ["latency", "memory_usage", "throughput"],
}

def get_phase_config(phase: int) -> Dict[str, Any]:
    """Get configuration for a specific experimental phase.
    
    Args:
        phase: Phase number (1, 2, or 3)
        
    Returns:
        Configuration dictionary for the specified phase
    """
    base_config = DEFAULT_EXPERIMENT_CONFIG.copy()
    
    if phase == 1:
        # Phase 1: Original implementation verification
        return {
            **base_config,
            "model_config": ORIGINAL_MODEL_CONFIGS,
            "embedding_model": EMBEDDING_MODELS["default"],
        }
    elif phase == 2:
        # Phase 2: Qwen2.5 Coder integration
        return {
            **base_config,
            "model_config": OLLAMA_CONFIG,
            "embedding_model": EMBEDDING_MODELS["default"],
        }
    elif phase == 3:
        # Phase 3: XnX notation integration
        return {
            **base_config,
            "model_configs": [ORIGINAL_MODEL_CONFIGS, OLLAMA_CONFIG],
            "embedding_model": EMBEDDING_MODELS["default"],
            "xnx_config": XNX_CONFIGS["default"],
        }
    else:
        raise ValueError(f"Invalid phase number: {phase}")

def load_env_vars():
    """Load environment variables for sensitive configuration."""
    # Database credentials
    if os.environ.get("ARANGO_PASSWORD"):
        ARANGO_CONFIG["password"] = os.environ["ARANGO_PASSWORD"]
    
    if os.environ.get("NEO4J_PASSWORD"):
        NEO4J_CONFIG["password"] = os.environ["NEO4J_PASSWORD"]
    
    # API keys for external services
    if os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

# Initialize with environment variables
load_env_vars()
