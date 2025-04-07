# Subgraph Extractor Module Documentation

## Overview

The Subgraph Extractor module is a critical component of the GraphRAG framework that extracts relevant subgraphs from knowledge graphs based on query context. It provides a comprehensive set of tools for scoring, pruning, constraining, and optimizing subgraphs to enhance retrieval quality.

## Components

The module consists of five main components:

1. **NodeEdgeScorer**: Scores the relevance of nodes and edges in a knowledge graph
2. **ContextPruner**: Provides context-aware pruning algorithms to remove less relevant parts
3. **SizeConstrainer**: Constrains subgraphs to manageable sizes while maximizing relevance
4. **SubgraphOptimizer**: Optimizes extracted subgraphs for relevance, connectivity, and diversity
5. **SubgraphExtractor**: Main interface that integrates all components

## Configuration System

The Subgraph Extractor module uses a flexible configuration system that supports:

- In-code configuration with default parameters
- Configuration via YAML or JSON files
- Predefined configuration presets for common use cases
- Component-specific configurations

## Configuration Options

### Main Extractor Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_nodes` | Maximum number of nodes in extracted subgraph | 100 |
| `max_edges` | Maximum number of edges in extracted subgraph | 500 |
| `max_density` | Maximum edge density (edges/max_possible_edges) | 0.1 |
| `relevance_threshold` | Minimum relevance score for nodes/edges | 0.3 |
| `context_aware` | Whether to use context-aware pruning | True |
| `optimize_subgraph` | Whether to apply optimization techniques | True |
| `embedding_attr` | Attribute name for node/edge embeddings | 'embedding' |
| `text_attr` | Attribute name for node/edge text content | 'text' |
| `preserve_seed_nodes` | Whether to always keep seed nodes | True |
| `relevance_weight` | Weight for relevance in hybrid optimization | 0.6 |
| `diversity_weight` | Weight for diversity in hybrid optimization | 0.2 |
| `connectivity_weight` | Weight for connectivity in hybrid optimization | 0.2 |

### NodeEdgeScorer Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `embedding_attr` | Attribute name for embeddings | 'embedding' |
| `text_attr` | Attribute name for text content | 'text' |
| `importance_attr` | Attribute name for predefined importance | 'importance' |
| `alpha` | Weight for semantic similarity in scoring | 0.5 |
| `beta` | Weight for structural importance in scoring | 0.3 |
| `gamma` | Weight for predefined importance in scoring | 0.2 |
| `min_score` | Minimum score threshold (scores below become 0) | 0.0 |
| `normalize_scores` | Whether to normalize scores to [0,1] range | True |

### ContextPruner Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `node_threshold` | Threshold for node relevance score | 0.3 |
| `edge_threshold` | Threshold for edge relevance score | 0.2 |
| `preserve_connectivity` | Whether to preserve connectivity in pruned graph | True |
| `preserve_seed_nodes` | Whether to always keep seed nodes | True |
| `preserve_edge_weight_attr` | Attribute name for edge weights | 'weight' |
| `max_edge_distance` | Maximum distance for preserving connectivity | 3 |

### SizeConstrainer Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_nodes` | Maximum number of nodes | 100 |
| `max_edges` | Maximum number of edges | 500 |
| `max_density` | Maximum edge density | 0.1 |
| `prioritize_by` | Method for prioritizing nodes ('relevance', 'degree', 'connectivity') | 'relevance' |
| `preserve_seed_nodes` | Whether to always keep seed nodes | True |
| `balance_threshold` | Balance between relevance and connectivity (0-1) | 0.5 |

### SubgraphOptimizer Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `relevance_weight` | Weight for relevance score in optimization | 0.6 |
| `diversity_weight` | Weight for diversity score in optimization | 0.2 |
| `connectivity_weight` | Weight for connectivity score in optimization | 0.2 |
| `max_iterations` | Maximum iterations for optimization algorithms | 10 |
| `improvement_threshold` | Minimum improvement to continue optimization | 0.01 |
| `random_seed` | Random seed for reproducibility (null = no seed) | None |

## Configuration Methods

### Method 1: Default Configuration

```python
from implementations.graphrag.core.subgraph_extractor import SubgraphExtractor

# Use default configuration
extractor = SubgraphExtractor()
```

### Method 2: Using Configuration Presets

```python
from implementations.graphrag.core.subgraph_extractor import SubgraphExtractor
from implementations.graphrag.core.subgraph_extractor.config import ExtractorPresets

# Use the large graph preset
extractor = SubgraphExtractor(ExtractorPresets.large_graph())

# Other available presets:
# - ExtractorPresets.small_graph()
# - ExtractorPresets.precision_focused()
# - ExtractorPresets.recall_focused()
# - ExtractorPresets.connectivity_focused()
```

### Method 3: Custom Configuration in Code

```python
from implementations.graphrag.core.subgraph_extractor import SubgraphExtractor
from implementations.graphrag.core.subgraph_extractor.config import SubgraphExtractorConfig

# Create custom configuration
custom_config = SubgraphExtractorConfig(
    max_nodes=150,
    relevance_threshold=0.4,
    optimize_subgraph=True,
    # Configure component-specific settings
    scorer_config=NodeEdgeScorerConfig(
        alpha=0.7,  # More weight to semantic similarity
        beta=0.2,
        gamma=0.1
    )
)

# Create extractor with custom config
extractor = SubgraphExtractor(custom_config)
```

### Method 4: Configuration from File

```python
from implementations.graphrag.core.subgraph_extractor import SubgraphExtractor
from implementations.graphrag.core.subgraph_extractor.config import load_config_from_file

# Load configuration from YAML or JSON file
config = load_config_from_file('path/to/config.yaml')

# Create extractor with loaded config
extractor = SubgraphExtractor(config)
```

## Creating and Saving Configurations

You can create a configuration and save it to a file for reuse:

```python
from implementations.graphrag.core.subgraph_extractor.config import (
    SubgraphExtractorConfig, save_config_to_file
)

# Create configuration
config = SubgraphExtractorConfig(
    max_nodes=200,
    relevance_threshold=0.4
)

# Save to file (YAML or JSON)
save_config_to_file(config, 'path/to/output_config.yaml')
```

## Configuration Presets

The module provides several predefined configuration presets:

### Small Graph Preset

Optimized for smaller graphs with fewer nodes and edges:

```python
small_graph_config = ExtractorPresets.small_graph()
# max_nodes = 50, max_edges = 100
```

### Large Graph Preset

Optimized for larger graphs with more aggressive filtering:

```python
large_graph_config = ExtractorPresets.large_graph()
# max_nodes = 200, max_edges = 1000, relevance_threshold = 0.4
```

### Precision-Focused Preset

Optimized for high precision with stricter relevance filtering:

```python
precision_config = ExtractorPresets.precision_focused()
# relevance_threshold = 0.5, alpha = 0.7 (more weight to semantic similarity)
```

### Recall-Focused Preset

Optimized for high recall with more lenient thresholds:

```python
recall_config = ExtractorPresets.recall_focused()
# relevance_threshold = 0.2, max_nodes = 150
```

### Connectivity-Focused Preset

Optimized for preserving connectivity between important nodes:

```python
connectivity_config = ExtractorPresets.connectivity_focused()
# connectivity_weight = 0.5, prioritize_by = 'connectivity'
```

## Best Practices

1. **Start with presets**: Begin with a preset configuration that matches your use case, then make incremental adjustments.

2. **Tune relevance_threshold**: This is the most important parameter for controlling the trade-off between precision and recall.

3. **Adjust for graph size**: Larger graphs need more aggressive pruning with higher thresholds.

4. **Balance weights**: The alpha, beta, and gamma weights control the balance between semantic, structural, and predefined importance.

5. **Preserve connectivity**: Keep preserve_connectivity enabled to ensure the extracted subgraph remains connected.

6. **Test and iterate**: Try different configurations on sample queries and adjust based on results.

## Examples

### Example 1: Basic Extraction

```python
import networkx as nx
import numpy as np
from implementations.graphrag.core.subgraph_extractor import SubgraphExtractor

# Create a graph
graph = nx.Graph()
# ... add nodes and edges ...

# Create an extractor with default configuration
extractor = SubgraphExtractor()

# Extract subgraph based on query embedding
query_embedding = np.array([0.1, 0.2, 0.3, 0.4])
seed_nodes = ["node1", "node2"]
subgraph, scores = extractor.extract_subgraph(
    graph, 
    query_embedding=query_embedding,
    seed_nodes=seed_nodes
)
```

### Example 2: Multi-Query Extraction

```python
# Create multiple query embeddings
query_embeddings = [
    np.array([0.1, 0.2, 0.3, 0.4]),
    np.array([0.2, 0.3, 0.4, 0.5])
]
seed_nodes_list = [["node1"], ["node2"]]

# Extract subgraph relevant to multiple queries
merged_subgraph = extractor.extract_multi_query_subgraph(
    graph,
    query_embeddings=query_embeddings,
    seed_nodes_list=seed_nodes_list,
    merge_method='weighted'
)
```

### Example 3: Custom Configuration for Specific Use Case

```python
from implementations.graphrag.core.subgraph_extractor.config import SubgraphExtractorConfig

# Create configuration for question answering
qa_config = SubgraphExtractorConfig(
    max_nodes=150,
    relevance_threshold=0.35,
    scorer_config=NodeEdgeScorerConfig(
        alpha=0.7,  # Emphasize semantic similarity
        beta=0.2,   # Some weight to structural importance
        gamma=0.1   # Less weight to predefined importance
    ),
    pruner_config=ContextPrunerConfig(
        node_threshold=0.35,
        edge_threshold=0.25
    )
)

# Create extractor with QA configuration
qa_extractor = SubgraphExtractor(qa_config)
```
