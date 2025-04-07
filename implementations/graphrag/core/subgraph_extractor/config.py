"""
Configuration module for the Subgraph Extractor.

This module provides configuration classes and helper functions for the Subgraph Extractor
components, allowing for easier configuration through both code and configuration files.
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

@dataclass
class NodeEdgeScorerConfig:
    """Configuration settings for the NodeEdgeScorer component."""
    
    # Attribute names
    embedding_attr: str = 'embedding'
    text_attr: str = 'text'
    importance_attr: str = 'importance'
    
    # Scoring weights
    alpha: float = 0.5  # Weight for semantic similarity
    beta: float = 0.3   # Weight for structural importance
    gamma: float = 0.2  # Weight for predefined importance
    
    # Thresholds and normalization
    min_score: float = 0.0
    normalize_scores: bool = True

@dataclass
class ContextPrunerConfig:
    """Configuration settings for the ContextPruner component."""
    
    # Relevance thresholds
    node_threshold: float = 0.3
    edge_threshold: float = 0.2
    
    # Connectivity settings
    preserve_connectivity: bool = True
    preserve_seed_nodes: bool = True
    preserve_edge_weight_attr: Optional[str] = 'weight'
    max_edge_distance: int = 3

@dataclass
class SizeConstrainerConfig:
    """Configuration settings for the SizeConstrainer component."""
    
    # Size constraints
    max_nodes: int = 100
    max_edges: int = 500
    max_density: float = 0.1
    
    # Prioritization settings
    prioritize_by: str = 'relevance'  # 'relevance', 'degree', 'connectivity'
    preserve_seed_nodes: bool = True
    balance_threshold: float = 0.5  # Balance between relevance and connectivity

@dataclass
class SubgraphOptimizerConfig:
    """Configuration settings for the SubgraphOptimizer component."""
    
    # Optimization weights
    relevance_weight: float = 0.6
    diversity_weight: float = 0.2
    connectivity_weight: float = 0.2
    
    # Optimization parameters
    max_iterations: int = 10
    improvement_threshold: float = 0.01
    random_seed: Optional[int] = None

@dataclass
class SubgraphExtractorConfig:
    """Configuration settings for the main SubgraphExtractor component."""
    
    # Size and density limits
    max_nodes: int = 100
    max_edges: int = 500
    max_density: float = 0.1
    
    # General extraction settings
    relevance_threshold: float = 0.3
    context_aware: bool = True
    optimize_subgraph: bool = True
    
    # Attribute names
    embedding_attr: str = 'embedding'
    text_attr: str = 'text'
    
    # Preservation settings
    preserve_seed_nodes: bool = True
    
    # Weight distribution
    relevance_weight: float = 0.6
    diversity_weight: float = 0.2
    connectivity_weight: float = 0.2
    
    # Component-specific configurations
    scorer_config: NodeEdgeScorerConfig = field(default_factory=NodeEdgeScorerConfig)
    pruner_config: ContextPrunerConfig = field(default_factory=ContextPrunerConfig)
    constrainer_config: SizeConstrainerConfig = field(default_factory=SizeConstrainerConfig)
    optimizer_config: SubgraphOptimizerConfig = field(default_factory=SubgraphOptimizerConfig)

@dataclass
class ExtractorPresets:
    """Predefined presets for different use cases."""
    
    @staticmethod
    def small_graph() -> SubgraphExtractorConfig:
        """Configuration preset for small graphs."""
        config = SubgraphExtractorConfig()
        config.max_nodes = 50
        config.max_edges = 100
        config.constrainer_config.max_nodes = 50
        config.constrainer_config.max_edges = 100
        return config
    
    @staticmethod
    def large_graph() -> SubgraphExtractorConfig:
        """Configuration preset for large graphs."""
        config = SubgraphExtractorConfig()
        config.max_nodes = 200
        config.max_edges = 1000
        config.constrainer_config.max_nodes = 200
        config.constrainer_config.max_edges = 1000
        config.relevance_threshold = 0.4  # More aggressive filtering
        config.pruner_config.node_threshold = 0.4
        return config
    
    @staticmethod
    def precision_focused() -> SubgraphExtractorConfig:
        """Configuration preset focusing on precision."""
        config = SubgraphExtractorConfig()
        config.relevance_threshold = 0.5  # Higher threshold for relevance
        config.pruner_config.node_threshold = 0.5
        config.pruner_config.edge_threshold = 0.4
        config.scorer_config.alpha = 0.7  # More weight to semantic similarity
        config.scorer_config.beta = 0.2
        config.scorer_config.gamma = 0.1
        return config
    
    @staticmethod
    def recall_focused() -> SubgraphExtractorConfig:
        """Configuration preset focusing on recall."""
        config = SubgraphExtractorConfig()
        config.relevance_threshold = 0.2  # Lower threshold for relevance
        config.pruner_config.node_threshold = 0.2
        config.pruner_config.edge_threshold = 0.15
        config.max_nodes = 150  # Allow more nodes
        config.max_edges = 750
        config.constrainer_config.max_nodes = 150
        config.constrainer_config.max_edges = 750
        return config
    
    @staticmethod
    def connectivity_focused() -> SubgraphExtractorConfig:
        """Configuration preset focusing on connectivity."""
        config = SubgraphExtractorConfig()
        config.relevance_weight = 0.4
        config.connectivity_weight = 0.5  # More weight to connectivity
        config.diversity_weight = 0.1
        config.optimizer_config.connectivity_weight = 0.5
        config.optimizer_config.relevance_weight = 0.4
        config.optimizer_config.diversity_weight = 0.1
        config.constrainer_config.prioritize_by = 'connectivity'
        return config


def load_config_from_file(file_path: str) -> SubgraphExtractorConfig:
    """
    Load extractor configuration from a JSON or YAML file.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        SubgraphExtractorConfig: Loaded configuration
    """
    if not os.path.exists(file_path):
        logger.error(f"Configuration file not found: {file_path}")
        return SubgraphExtractorConfig()
    
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                config_dict = json.load(f)
            elif file_path.endswith(('.yaml', '.yml')):
                config_dict = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return SubgraphExtractorConfig()
        
        # Create main config
        main_config = SubgraphExtractorConfig()
        
        # Update main config fields
        for key, value in config_dict.items():
            if key not in ('scorer_config', 'pruner_config', 'constrainer_config', 'optimizer_config'):
                if hasattr(main_config, key):
                    setattr(main_config, key, value)
        
        # Update component configs if present
        if 'scorer_config' in config_dict:
            for key, value in config_dict['scorer_config'].items():
                if hasattr(main_config.scorer_config, key):
                    setattr(main_config.scorer_config, key, value)
                    
        if 'pruner_config' in config_dict:
            for key, value in config_dict['pruner_config'].items():
                if hasattr(main_config.pruner_config, key):
                    setattr(main_config.pruner_config, key, value)
                    
        if 'constrainer_config' in config_dict:
            for key, value in config_dict['constrainer_config'].items():
                if hasattr(main_config.constrainer_config, key):
                    setattr(main_config.constrainer_config, key, value)
                    
        if 'optimizer_config' in config_dict:
            for key, value in config_dict['optimizer_config'].items():
                if hasattr(main_config.optimizer_config, key):
                    setattr(main_config.optimizer_config, key, value)
        
        return main_config
        
    except Exception as e:
        logger.error(f"Error loading configuration from {file_path}: {str(e)}")
        return SubgraphExtractorConfig()


def save_config_to_file(config: SubgraphExtractorConfig, file_path: str) -> bool:
    """
    Save extractor configuration to a JSON or YAML file.
    
    Args:
        config: Configuration to save
        file_path: Path to save configuration
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert dataclass to dict
        config_dict = asdict(config)
        
        with open(file_path, 'w') as f:
            if file_path.endswith('.json'):
                json.dump(config_dict, f, indent=2)
            elif file_path.endswith(('.yaml', '.yml')):
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return False
        
        logger.info(f"Configuration saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving configuration to {file_path}: {str(e)}")
        return False
