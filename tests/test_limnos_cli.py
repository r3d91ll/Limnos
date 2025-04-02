#!/usr/bin/env python3
"""
Test script for the HADES CLI and modular architecture.

This script tests the basic functionality of the HADES modular architecture,
including component registration, configuration loading, and the ingest pipeline.
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Import HADES modules
from limnos.pipeline import config, registry
from limnos.pipeline.init import register_built_in_components
from limnos.ingest.pipeline import ModularIngestPipeline
from limnos.storage.redis.storage import RedisStorageBackend, RedisDocumentStore, RedisChunkStore, RedisEmbeddingStore


def test_component_registration():
    """Test component registration with the plugin registry."""
    logging.info("Testing component registration...")
    
    # Register built-in components
    registered = register_built_in_components()
    
    # Verify registration
    assert len(registered) > 0, "No components were registered"
    logging.info(f"Successfully registered {len(registered)} components")
    
    # Test component retrieval
    document_store_cls = registry.get_plugin('document_store', 'redis')
    assert document_store_cls is not None, "Failed to retrieve document_store.redis plugin"
    assert document_store_cls == RedisDocumentStore, "Retrieved wrong class for document_store.redis"
    
    chunk_store_cls = registry.get_plugin('chunk_store', 'redis')
    assert chunk_store_cls is not None, "Failed to retrieve chunk_store.redis plugin"
    assert chunk_store_cls == RedisChunkStore, "Retrieved wrong class for chunk_store.redis"
    
    embedding_store_cls = registry.get_plugin('embedding_store', 'redis')
    assert embedding_store_cls is not None, "Failed to retrieve embedding_store.redis plugin"
    assert embedding_store_cls == RedisEmbeddingStore, "Retrieved wrong class for embedding_store.redis"
    
    logging.info("Component registration test passed!")
    return True


def test_configuration_loading():
    """Test configuration loading from a file."""
    logging.info("Testing configuration loading...")
    
    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_file.write("""
global:
  debug: true
  environment: test

storage:
  redis:
    host: localhost
    port: 6379
    db: 0
  
  document_store:
    type: redis
    prefix: doc:
  
  chunk_store:
    type: redis
    prefix: chunk:
  
  embedding_store:
    type: redis
    prefix: emb:

models:
  embedding_model:
    type: openai
    model_name: text-embedding-3-small
    dimensions: 1536
    api_key: ${OPENAI_API_KEY}

ingest:
  document_processor:
    type: basic
    extract_metadata: true
  
  chunker:
    type: text
    chunk_size: 1000
    chunk_overlap: 200
    chunking_strategy: paragraph
        """)
        config_path = temp_file.name
    
    try:
        # Load configuration
        config.load_config(config_path)
        
        # Verify configuration
        assert config.get('global.debug') is True, "Failed to load global.debug"
        assert config.get('storage.redis.host') == 'localhost', "Failed to load storage.redis.host"
        assert config.get('models.embedding_model.model_name') == 'text-embedding-3-small', "Failed to load models.embedding_model.model_name"
        
        logging.info("Configuration loading test passed!")
        return True
    finally:
        # Clean up
        os.unlink(config_path)


def test_redis_connection():
    """Test connection to Redis."""
    logging.info("Testing Redis connection...")
    
    # Initialize Redis backend
    redis_config = {
        'host': 'localhost',
        'port': 6379,
        'db': 0
    }
    
    backend = RedisStorageBackend()
    backend.initialize(redis_config)
    
    # Test connection
    connected = backend.connect()
    if connected:
        logging.info("Successfully connected to Redis")
    else:
        logging.warning("Failed to connect to Redis - this test is skipped but not failed as Redis might not be running")
    
    return connected


def test_ingest_pipeline():
    """Test the ingest pipeline with a simple text document."""
    logging.info("Testing ingest pipeline...")
    
    # Skip if Redis is not available
    if not test_redis_connection():
        logging.warning("Skipping ingest pipeline test as Redis is not available")
        return None
    
    # Create a simple configuration
    pipeline_config = {
        "document_processor": {
            "type": "basic",
            "extract_metadata": True
        },
        "chunker": {
            "type": "text",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "chunking_strategy": "paragraph"
        },
        "embedding_model": {
            "type": "local",
            "model_name": "all-MiniLM-L6-v2",
            "dimensions": 384,
            "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            "batch_size": 32
        },
        "document_store": {
            "type": "redis",
            "prefix": "test:doc:",
            "host": "localhost",
            "port": 6379,
            "db": 0
        },
        "chunk_store": {
            "type": "redis",
            "prefix": "test:chunk:",
            "host": "localhost",
            "port": 6379,
            "db": 0
        },
        "embedding_store": {
            "type": "redis",
            "prefix": "test:emb:",
            "host": "localhost",
            "port": 6379,
            "db": 0
        }
    }
    
    # Check if we're using the OpenAI model and if the API key is available
    if pipeline_config["embedding_model"].get("type") == "openai" and not os.environ.get("OPENAI_API_KEY"):
        logging.warning("Skipping embedding generation as OPENAI_API_KEY is not set for OpenAI model")
        return None
    
    # Initialize pipeline
    pipeline = ModularIngestPipeline()
    pipeline.initialize(pipeline_config)
    
    # Create a test document
    test_text = """
    # HADES Modular Architecture
    
    The HADES (Hierarchical Actor-based Document Embedding System) architecture provides a modular, 
    extensible framework for building and experimenting with retrieval-augmented generation systems. 
    
    ## Key Features
    
    - Modular Pipeline Architecture
    - Plugin System
    - Configurable Components
    - Redis Integration
    - Actor-Network Theory
    
    This document is a test for the ingest pipeline.
    """
    
    # Process the document
    document, chunks, embedded_chunks = pipeline.process_text(test_text)
    
    # Verify results
    assert document is not None, "Failed to create document"
    assert len(chunks) > 0, "Failed to create chunks"
    assert len(embedded_chunks) > 0, "Failed to create embeddings"
    
    logging.info(f"Successfully processed document: {document.doc_id}")
    logging.info(f"Created {len(chunks)} chunks")
    logging.info(f"Generated {len(embedded_chunks)} embeddings")
    
    # Clean up (optional)
    document_store = RedisDocumentStore()
    document_store.initialize(pipeline_config["document_store"])
    document_store.connect()
    document_store.delete(document.doc_id)
    
    chunk_store = RedisChunkStore()
    chunk_store.initialize(pipeline_config["chunk_store"])
    chunk_store.connect()
    for chunk in chunks:
        chunk_store.delete(chunk.chunk_id)
    
    embedding_store = RedisEmbeddingStore()
    embedding_store.initialize(pipeline_config["embedding_store"])
    embedding_store.connect()
    for embedded_chunk in embedded_chunks:
        embedding_store.delete(embedded_chunk.chunk_id)
    
    logging.info("Ingest pipeline test passed!")
    return True


def main():
    """Run all tests."""
    tests = [
        test_component_registration,
        test_configuration_loading,
        test_redis_connection,
        test_ingest_pipeline
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            logging.error(f"Error in {test_func.__name__}: {e}")
            results.append((test_func.__name__, False))
    
    # Print summary
    logging.info("\n=== Test Summary ===")
    all_passed = True
    for name, result in results:
        status = "PASSED" if result else "FAILED" if result is False else "SKIPPED"
        if result is False:
            all_passed = False
        logging.info(f"{name}: {status}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
