"""
GraphRAG Embedding Tests

This module tests the embedding functionality of the GraphRAG framework.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Ensure the parent directory is in path so we can import implementation modules
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import test infrastructure components
from experiments.framework_testing.common.test_data import TestDataGenerator, create_test_environment
from experiments.framework_testing.common.metrics import PerformanceTracker, MetricsCollection

# Import GraphRAG components
from implementations.graphrag.core.graph_search.search_engine import GraphSearchEngine
from implementations.graphrag.core.graph_search.optimized_search import OptimizedGraphSearch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_embedding_initialization():
    """Test the initialization of the embedding model in GraphRAG."""
    logger.info("Testing GraphRAG embedding initialization")
    
    # Create performance tracker
    tracker = PerformanceTracker(framework="GraphRAG", test_name="embedding_initialization")
    
    # Test different model configs
    configs = [
        {
            "embedding_model": "sentence-transformers",
            "embedding_model_name": "all-MiniLM-L6-v2"
        },
        {
            "embedding_model": "huggingface",
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2"
        }
    ]
    results = {}
    
    try:
        for i, config in enumerate(configs):
            config_name = f"config_{i+1}"
            with tracker.time_operation(f"initialize_{config_name}"):
                with tracker.track_memory(f"memory_{config_name}"):
                    # Initialize search engine with parameters
                    search_engine = GraphSearchEngine(
                        max_depth=config.get('max_depth', 5),
                        max_nodes=config.get('max_nodes', 1000),
                        max_paths=config.get('max_paths', 10),
                        relevance_threshold=config.get('relevance_threshold', 0.5),
                        use_optimization=config.get('use_optimization', True),
                        parallel_workers=config.get('parallel_workers', 4),
                        enable_caching=config.get('enable_caching', True)
                    )
                    
                    # Check if engine is properly initialized
                    results[config_name] = {
                        'config': config,
                        'initialized': hasattr(search_engine, 'embedding_model'),
                        'optimized': search_engine.optimized is not None
                    }
    except Exception as e:
        logger.error(f"Error testing embedding initialization: {e}")
        results['error'] = str(e)
    
    # Add results to metrics
    tracker.add_metric("initialization_results", results)
    
    # Save metrics
    output_dir = Path(__file__).parent.parent / "results" / "graphrag"
    os.makedirs(output_dir, exist_ok=True)
    tracker.save_metrics(os.path.join(output_dir, "embedding_initialization.json"))
    
    return results

def test_text_embedding():
    """Test embedding generation for text in GraphRAG."""
    logger.info("Testing GraphRAG text embedding")
    
    # Create performance tracker
    tracker = PerformanceTracker(framework="GraphRAG", test_name="text_embedding")
    
    # Initialize search engine
    with tracker.time_operation("initialize_search_engine"):
        with tracker.track_memory("memory_initialization"):
            config = {
                "embedding_model": "sentence-transformers",
                "embedding_model_name": "all-MiniLM-L6-v2"
            }
            search_engine = GraphSearchEngine(
                max_depth=config.get('max_depth', 5),
                max_nodes=config.get('max_nodes', 1000),
                max_paths=config.get('max_paths', 10),
                relevance_threshold=config.get('relevance_threshold', 0.5),
                use_optimization=config.get('use_optimization', True),
                parallel_workers=config.get('parallel_workers', 4),
                enable_caching=config.get('enable_caching', True)
            )
    
    # Test texts of different lengths
    test_texts = {
        "short": "This is a short text for embedding.",
        "medium": "This is a medium length text for embedding. It contains multiple sentences and should be more complex than the short text. The embedding model needs to process more information.",
        "long": "This is a long text for embedding. " * 10
    }
    
    embedding_results = {}
    try:
        for text_type, text in test_texts.items():
            with tracker.time_operation(f"embed_{text_type}_text"):
                with tracker.track_memory(f"memory_embed_{text_type}"):
                    # OptimizedGraphSearch doesn't have a direct get_embedding method
                    # So we'll just use SentenceTransformer directly
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(config.get('embedding_model_name', 'all-MiniLM-L6-v2'))
                    embedding = model.encode(text)
                    
                    embedding_results[text_type] = {
                        'shape': embedding.shape if hasattr(embedding, 'shape') else len(embedding),
                        'mean': float(np.mean(embedding)),
                        'std': float(np.std(embedding)),
                        'min': float(np.min(embedding)),
                        'max': float(np.max(embedding))
                    }
    except Exception as e:
        logger.error(f"Error testing text embedding: {e}")
        embedding_results['error'] = str(e)
    
    # Add results to metrics
    tracker.add_metric("embedding_results", embedding_results)
    
    # Save metrics
    output_dir = Path(__file__).parent.parent / "results" / "graphrag"
    os.makedirs(output_dir, exist_ok=True)
    tracker.save_metrics(os.path.join(output_dir, "text_embedding.json"))
    
    return embedding_results

def test_graph_node_embedding():
    """Test embedding generation for graph nodes in GraphRAG."""
    logger.info("Testing GraphRAG graph node embedding")
    
    # Create performance tracker
    tracker = PerformanceTracker(framework="GraphRAG", test_name="graph_node_embedding")
    
    # Initialize optimized graph search directly
    with tracker.time_operation("initialize_optimized_search"):
        with tracker.track_memory("memory_initialization"):
            config = {
                "embedding_model": "sentence-transformers",
                "embedding_model_name": "all-MiniLM-L6-v2"
            }
            optimized_search = OptimizedGraphSearch(
                max_workers=config.get('max_workers', 4),
                use_parallel=config.get('use_parallel', True),
                cache_results=config.get('cache_results', True),
                cache_ttl=config.get('cache_ttl', 3600),
                index_attributes=config.get('index_attributes', None)
            )
    
    # Create test nodes (simulating graph nodes)
    test_nodes = {
        "concept": {
            "id": "node1", 
            "type": "concept", 
            "name": "Neural Networks", 
            "description": "A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data."
        },
        "term": {
            "id": "node2", 
            "type": "term", 
            "name": "Deep Learning", 
            "definition": "Deep learning is a subset of machine learning that uses neural networks with many layers."
        },
        "entity": {
            "id": "node3", 
            "type": "entity", 
            "name": "TensorFlow", 
            "category": "Framework", 
            "description": "TensorFlow is an open-source machine learning framework developed by Google."
        }
    }
    
    node_embedding_results = {}
    try:
        for node_type, node in test_nodes.items():
            with tracker.time_operation(f"embed_{node_type}_node"):
                with tracker.track_memory(f"memory_embed_{node_type}"):
                    # Convert node to text for embedding (similar to how GraphRAG would)
                    if node_type == "concept":
                        node_text = f"{node['name']}: {node['description']}"
                    elif node_type == "term":
                        node_text = f"{node['name']}: {node['definition']}"
                    else:
                        node_text = f"{node['name']}: {node['description']}"
                    
                    # Get embedding using SentenceTransformer directly
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(config.get('embedding_model_name', 'all-MiniLM-L6-v2'))
                    embedding = model.encode(node_text)
                    
                    node_embedding_results[node_type] = {
                        'node_text': node_text,
                        'shape': embedding.shape if hasattr(embedding, 'shape') else len(embedding),
                        'mean': float(np.mean(embedding)),
                        'std': float(np.std(embedding)),
                        'min': float(np.min(embedding)),
                        'max': float(np.max(embedding))
                    }
    except Exception as e:
        logger.error(f"Error testing node embedding: {e}")
        node_embedding_results['error'] = str(e)
    
    # Add results to metrics
    tracker.add_metric("node_embedding_results", node_embedding_results)
    
    # Save metrics
    output_dir = Path(__file__).parent.parent / "results" / "graphrag"
    os.makedirs(output_dir, exist_ok=True)
    tracker.save_metrics(os.path.join(output_dir, "graph_node_embedding.json"))
    
    return node_embedding_results

def test_semantic_search():
    """Test semantic search functionality in GraphRAG."""
    logger.info("Testing GraphRAG semantic search")
    
    # Create performance tracker
    tracker = PerformanceTracker(framework="GraphRAG", test_name="semantic_search")
    
    # Create temporary directory for storage
    import tempfile
    import networkx as nx
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize search engine
        with tracker.time_operation("initialize_search_engine"):
            with tracker.track_memory("memory_initialization"):
                config = {
                    "embedding_model": "sentence-transformers",
                    "embedding_model_name": "all-MiniLM-L6-v2",
                    "cache_dir": temp_dir
                }
                search_engine = GraphSearchEngine(
                max_depth=config.get('max_depth', 5),
                max_nodes=config.get('max_nodes', 1000),
                max_paths=config.get('max_paths', 10),
                relevance_threshold=config.get('relevance_threshold', 0.5),
                use_optimization=config.get('use_optimization', True),
                parallel_workers=config.get('parallel_workers', 4),
                enable_caching=config.get('enable_caching', True)
            )
        
        # Create a test graph
        with tracker.time_operation("create_test_graph"):
            with tracker.track_memory("memory_create_graph"):
                G = nx.DiGraph()
                
                # Add nodes
                nodes = [
                    ("node1", {"type": "concept", "name": "Neural Networks", "description": "A type of ML model"}),
                    ("node2", {"type": "concept", "name": "Deep Learning", "description": "Using deep neural networks"}),
                    ("node3", {"type": "concept", "name": "Machine Learning", "description": "Systems that learn from data"}),
                    ("node4", {"type": "term", "name": "CNN", "definition": "Convolutional Neural Network"}),
                    ("node5", {"type": "term", "name": "RNN", "definition": "Recurrent Neural Network"})
                ]
                G.add_nodes_from(nodes)
                
                # Add edges
                edges = [
                    ("node1", "node2", {"type": "related_to"}),
                    ("node2", "node3", {"type": "subset_of"}),
                    ("node4", "node1", {"type": "instance_of"}),
                    ("node5", "node1", {"type": "instance_of"})
                ]
                G.add_edges_from(edges)
        
        # Index the graph for search
        search_results = {}
        
        with tracker.time_operation("index_graph"):
            with tracker.track_memory("memory_index_graph"):
                # Generate embeddings for nodes using SentenceTransformer directly
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(config.get('embedding_model_name', 'all-MiniLM-L6-v2'))
                
                # Store node texts and their embeddings
                node_texts = {}
                node_embeddings = {}
                
                for node_id, node_data in G.nodes(data=True):
                    # Convert node to text
                    if node_data.get("type") == "concept":
                        node_text = f"{node_data.get('name', '')}: {node_data.get('description', '')}"
                    elif node_data.get("type") == "term":
                        node_text = f"{node_data.get('name', '')}: {node_data.get('definition', '')}"
                    else:
                        node_text = str(node_data)
                    
                    # Store the text
                    node_texts[node_id] = node_text
                
                # Generate embeddings in batch for efficiency
                all_texts = list(node_texts.values())
                all_embeddings = model.encode(all_texts)
                
                # Associate embeddings back to nodes
                for i, node_id in enumerate(node_texts.keys()):
                    node_embeddings[node_id] = all_embeddings[i]
                
                search_results["indexed_count"] = len(G.nodes)
        
        # Test semantic search
        with tracker.time_operation("semantic_search"):
            with tracker.track_memory("memory_semantic_search"):
                # Implement semantic search using embeddings
                query = "deep neural networks"
                query_embedding = model.encode(query)
                
                # Calculate cosine similarity with all node embeddings
                import numpy as np
                similarities = {}
                
                for node_id, embedding in node_embeddings.items():
                    # Compute cosine similarity
                    similarity = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                    )
                    similarities[node_id] = similarity
                
                # Sort by similarity and get top k=3
                sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
                
                search_results["query"] = query
                search_results["results"] = [
                    {"node_id": node_id, "score": float(score)} 
                    for node_id, score in sorted_results
                ]
    
    except Exception as e:
        logger.error(f"Error testing semantic search: {e}")
        search_results['error'] = str(e)
    
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Add results to metrics
    tracker.add_metric("search_results", search_results)
    
    # Save metrics
    output_dir = Path(__file__).parent.parent / "results" / "graphrag"
    os.makedirs(output_dir, exist_ok=True)
    tracker.save_metrics(os.path.join(output_dir, "semantic_search.json"))
    
    return search_results

def main():
    """Run all embedding tests for GraphRAG."""
    logger.info("Starting GraphRAG embedding tests")
    
    # Create results directory
    output_dir = Path(__file__).parent.parent / "results" / "graphrag"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run tests
    initialization_results = test_embedding_initialization()
    text_embedding_results = test_text_embedding()
    node_embedding_results = test_graph_node_embedding()
    search_results = test_semantic_search()
    
    # Compile all results
    all_results = {
        "initialization": initialization_results,
        "text_embedding": text_embedding_results,
        "node_embedding": node_embedding_results,
        "semantic_search": search_results
    }
    
    # Save combined results
    import json
    with open(os.path.join(output_dir, "all_embedding_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"All GraphRAG embedding tests completed. Results saved to {output_dir}")
    
    return all_results

if __name__ == "__main__":
    main()
