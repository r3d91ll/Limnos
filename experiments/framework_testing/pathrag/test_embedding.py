"""
PathRAG Embedding Tests

This module tests the embedding functionality of the PathRAG framework.
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

from experiments.framework_testing.common.test_data import TestDataGenerator, create_test_environment
from experiments.framework_testing.common.metrics import PerformanceTracker, MetricsCollection

# Import PathRAG components
# Import PathRAG components directly
from implementations.pathrag.core.path_vector_store import PathVectorStore
from implementations.pathrag.core.path_structures import Path as RagPath, PathNode, PathEdge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_embedding_initialization():
    """Test the initialization of the embedding model in PathVectorStore."""
    logger.info("Testing PathVectorStore embedding initialization")
    
    # Create performance tracker
    tracker = PerformanceTracker(framework="PathRAG", test_name="embedding_initialization")
    
    # Test local embedding options only - prioritizing hardware acceleration
    backends = ["sentence_transformers"]  # This already runs locally on CUDA
    # TODO: Add ollama as a backend option once implemented
    # TODO: Add vLLM as a backend option for better performance
    results = {}
    
    try:
        for backend in backends:
            with tracker.time_operation(f"initialize_{backend}"):
                with tracker.track_memory(f"memory_{backend}"):
                    # Initialize vector store with the backend
                    config = {
                        'model_backend': backend,
                        'model_name': 'all-MiniLM-L6-v2' if backend == 'sentence_transformers' else 'sentence-transformers/all-MiniLM-L6-v2',
                        'dimension': 384,
                        'normalize_embeddings': True
                    }
                    vector_store = PathVectorStore(config)
                    vector_store.initialize_model()
                    
                    # Check if model is initialized
                    results[backend] = {
                        'model_initialized': vector_store.model_initialized,
                        'model_type': str(type(vector_store.model))
                    }
    except Exception as e:
        logger.error(f"Error testing embedding initialization: {e}")
        results['error'] = str(e)
    
    # Add results to metrics
    tracker.add_metric("initialization_results", results)
    
    # Save metrics
    output_dir = Path(__file__).parent.parent / "results" / "pathrag"
    os.makedirs(output_dir, exist_ok=True)
    tracker.save_metrics(os.path.join(output_dir, "embedding_initialization.json"))
    
    return results

def test_text_embedding():
    """Test embedding generation for text in PathVectorStore."""
    logger.info("Testing PathVectorStore text embedding")
    
    # Create performance tracker
    tracker = PerformanceTracker(framework="PathRAG", test_name="text_embedding")
    
    # Initialize vector store
    with tracker.time_operation("initialize_vector_store"):
        with tracker.track_memory("memory_initialization"):
            config = {
                'model_backend': 'sentence_transformers',
                'model_name': 'all-MiniLM-L6-v2',
                'dimension': 384,
                'normalize_embeddings': True
            }
            vector_store = PathVectorStore(config)
            vector_store.initialize_model()
    
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
                    embedding = vector_store._get_embedding(text)
                    embedding_results[text_type] = {
                        'shape': embedding.shape,
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
    output_dir = Path(__file__).parent.parent / "results" / "pathrag"
    os.makedirs(output_dir, exist_ok=True)
    tracker.save_metrics(os.path.join(output_dir, "text_embedding.json"))
    
    return embedding_results

def test_path_embedding():
    """Test embedding generation for paths in PathVectorStore."""
    logger.info("Testing PathVectorStore path embedding")
    
    # Create performance tracker
    tracker = PerformanceTracker(framework="PathRAG", test_name="path_embedding")
    
    # Initialize vector store
    with tracker.time_operation("initialize_vector_store"):
        with tracker.track_memory("memory_initialization"):
            config = {
                'model_backend': 'sentence_transformers',
                'model_name': 'all-MiniLM-L6-v2',
                'dimension': 384,
                'normalize_embeddings': True
            }
            vector_store = PathVectorStore(config)
            vector_store.initialize_model()
    
    # Create test paths
    test_paths = {
        "short_path": RagPath(
            id="path1",
            nodes=[
                PathNode(id="node1", type="concept", text="Neural Networks"),
                PathNode(id="node2", type="concept", text="Deep Learning")
            ],
            edges=[
                PathEdge(id="edge1", source_id="node1", target_id="node2", type="is_related_to")
            ]
        ),
        "medium_path": RagPath(
            id="path2",
            nodes=[
                PathNode(id="node1", type="concept", text="Neural Networks"),
                PathNode(id="node2", type="concept", text="Deep Learning"),
                PathNode(id="node3", type="concept", text="Machine Learning"),
                PathNode(id="node4", type="concept", text="Artificial Intelligence")
            ],
            edges=[
                PathEdge(id="edge1", source_id="node1", target_id="node2", type="is_related_to"),
                PathEdge(id="edge2", source_id="node2", target_id="node3", type="is_subset_of"),
                PathEdge(id="edge3", source_id="node3", target_id="node4", type="is_subset_of")
            ]
        )
    }
    
    path_embedding_results = {}
    try:
        for path_type, path in test_paths.items():
            with tracker.time_operation(f"embed_{path_type}"):
                with tracker.track_memory(f"memory_embed_{path_type}"):
                    # Test path to text conversion
                    path_text = vector_store._path_to_text(path)
                    
                    # Test path embedding
                    embedding = vector_store.get_path_embedding(path)
                    
                    path_embedding_results[path_type] = {
                        'path_text': path_text,
                        'shape': embedding.shape,
                        'mean': float(np.mean(embedding)),
                        'std': float(np.std(embedding)),
                        'min': float(np.min(embedding)),
                        'max': float(np.max(embedding))
                    }
    except Exception as e:
        logger.error(f"Error testing path embedding: {e}")
        path_embedding_results['error'] = str(e)
    
    # Add results to metrics
    tracker.add_metric("path_embedding_results", path_embedding_results)
    
    # Save metrics
    output_dir = Path(__file__).parent.parent / "results" / "pathrag"
    os.makedirs(output_dir, exist_ok=True)
    tracker.save_metrics(os.path.join(output_dir, "path_embedding.json"))
    
    return path_embedding_results

def test_embedding_storage():
    """Test embedding storage and retrieval in PathVectorStore."""
    logger.info("Testing PathVectorStore embedding storage")
    
    # Create performance tracker
    tracker = PerformanceTracker(framework="PathRAG", test_name="embedding_storage")
    
    # Create temporary directory for storage
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize vector store with storage
        with tracker.time_operation("initialize_vector_store"):
            with tracker.track_memory("memory_initialization"):
                config = {
                    'model_backend': 'sentence_transformers',
                    'model_name': 'all-MiniLM-L6-v2',
                    'dimension': 384,
                    'normalize_embeddings': True,
                    'storage_dir': temp_dir
                }
                vector_store = PathVectorStore(config)
                vector_store.initialize_model()
        
        # Create test paths
        test_paths = {}
        for i in range(5):
            path = RagPath(
                id=f"path{i}",
                nodes=[
                    PathNode(id=f"node{i}1", type="concept", text=f"Concept {i}1"),
                    PathNode(id=f"node{i}2", type="concept", text=f"Concept {i}2")
                ],
                edges=[
                    PathEdge(id=f"edge{i}1", source_id=f"node{i}1", target_id=f"node{i}2", type="is_related_to")
                ]
            )
            test_paths[f"path{i}"] = path
        
        # Generate and store embeddings
        storage_results = {'store': {}, 'load': {}, 'retrieval': {}}
        
        with tracker.time_operation("store_embeddings"):
            with tracker.track_memory("memory_store"):
                for path_id, path in test_paths.items():
                    embedding = vector_store.get_path_embedding(path)
                    vector_store.add_embedding(path_id, embedding)
                
                # Save to disk
                save_path = vector_store.save()
                storage_results['store']['save_path'] = save_path
                storage_results['store']['num_embeddings'] = len(vector_store.embeddings)
        
        # Measure storage size
        tracker.measure_storage("embeddings_file", save_path)
        
        # Clear and reload
        with tracker.time_operation("clear_and_load"):
            with tracker.track_memory("memory_load"):
                # Clear the vector store
                vector_store.clear()
                
                # Verify it's empty
                storage_results['load']['after_clear'] = len(vector_store.embeddings)
                
                # Load from disk
                vector_store.load(save_path)
                storage_results['load']['after_load'] = len(vector_store.embeddings)
        
        # Test retrieval
        with tracker.time_operation("retrieval"):
            with tracker.track_memory("memory_retrieval"):
                # Test semantic search
                query = "concept related topics"
                search_results = vector_store.semantic_search(query, k=3)
                storage_results['retrieval']['semantic_search'] = [
                    {'path_id': path_id, 'score': float(score)} 
                    for path_id, score in search_results
                ]
                
                # Test similar paths
                if test_paths:
                    first_path_id = next(iter(test_paths.keys()))
                    similar = vector_store.similar_paths(first_path_id, k=3)
                    storage_results['retrieval']['similar_paths'] = [
                        {'path_id': path_id, 'score': float(score)} 
                        for path_id, score in similar
                    ]
    
    except Exception as e:
        logger.error(f"Error testing embedding storage: {e}")
        storage_results['error'] = str(e)
    
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Add results to metrics
    tracker.add_metric("storage_results", storage_results)
    
    # Save metrics
    output_dir = Path(__file__).parent.parent / "results" / "pathrag"
    os.makedirs(output_dir, exist_ok=True)
    tracker.save_metrics(os.path.join(output_dir, "embedding_storage.json"))
    
    return storage_results

def main():
    """Run all embedding tests for PathRAG."""
    logger.info("Starting PathRAG embedding tests")
    
    # Create results directory
    output_dir = Path(__file__).parent.parent / "results" / "pathrag"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run tests
    initialization_results = test_embedding_initialization()
    text_embedding_results = test_text_embedding()
    path_embedding_results = test_path_embedding()
    storage_results = test_embedding_storage()
    
    # Compile all results
    all_results = {
        "initialization": initialization_results,
        "text_embedding": text_embedding_results,
        "path_embedding": path_embedding_results,
        "storage": storage_results
    }
    
    # Save combined results
    import json
    with open(os.path.join(output_dir, "all_embedding_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"All PathRAG embedding tests completed. Results saved to {output_dir}")
    
    return all_results

if __name__ == "__main__":
    main()
