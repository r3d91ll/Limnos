#!/usr/bin/env python3
"""
Limnos CLI - Command Line Interface for the HADES modular pipeline architecture.

This script provides a command-line interface for interacting with the HADES system,
including dataset creation, embedding, retrieval, and inference.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from limnos.pipeline import config, registry
from limnos.ingest.pipeline import ModularIngestPipeline
from limnos.storage.redis.storage import RedisStorageBackend
# Import other storage backends as they become available
# from limnos.storage.disk.storage import DiskStorageBackend
# from limnos.storage.hybrid.storage import HybridStorageBackend


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


def load_configuration(config_path: Union[str, Path]) -> None:
    """Load configuration from a file or directory.
    
    Args:
        config_path: Path to configuration file or directory
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration path not found: {config_path}")
    
    # Load configuration
    config.load_config(config_path)
    
    # Load environment variables with HADES_ prefix
    config.get_config().load_environment_variables("HADES_")


def discover_plugins() -> None:
    """Discover and register plugins from the HADES package."""
    # Discover plugins in each module
    modules = [
        "hades.ingest",
        "hades.retrieve",
        "hades.infer",
        "hades.storage",
        "hades.models"
    ]
    
    for module in modules:
        try:
            count = registry.discover_plugins(module)
            logging.info(f"Discovered {count} plugins in {module}")
        except Exception as e:
            logging.error(f"Error discovering plugins in {module}: {e}")


def test_storage_connection() -> bool:
    """Test connection to the configured storage backend.
    
    Returns:
        True if connection successful, False otherwise
    """
    # Determine which storage backend to use
    storage_type = config.get("storage.type", "redis")
    
    if storage_type == "redis":
        redis_config = config.get("storage.redis", {})
        backend = RedisStorageBackend()
        backend.initialize(redis_config)
        return backend.connect()
    else:
        # For other storage types, we'll need to implement similar connection tests
        logging.warning(f"Connection test not implemented for storage type: {storage_type}")
        return True  # Assume success for now


def ingest_command(args: argparse.Namespace) -> None:
    """Handle the ingest command.
    
    Args:
        args: Command line arguments
    """
    # Initialize ingest pipeline
    pipeline_config = {
        "document_processor": config.get("ingest.document_processor", {}),
        "chunker": config.get("ingest.chunker", {}),
        "embedding_model": config.get("models.embedding_model", {}),
        "document_store": config.get("storage.document_store", {}),
        "chunk_store": config.get("storage.chunk_store", {}),
        "embedding_store": config.get("storage.embedding_store", {})
    }
    
    pipeline = ModularIngestPipeline()
    pipeline.initialize(pipeline_config)
    
    # Process input based on type
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logging.error(f"File not found: {file_path}")
            return
        
        logging.info(f"Processing file: {file_path}")
        document, chunks, embedded_chunks = pipeline.process_file(file_path)
        
        logging.info(f"Processed document: {document.doc_id}")
        logging.info(f"Created {len(chunks)} chunks")
        logging.info(f"Generated {len(embedded_chunks)} embeddings")
    
    elif args.directory:
        dir_path = Path(args.directory)
        if not dir_path.is_dir():
            logging.error(f"Directory not found: {dir_path}")
            return
        
        logging.info(f"Processing directory: {dir_path}")
        results = pipeline.process_directory(dir_path, recursive=args.recursive)
        
        logging.info(f"Processed {len(results)} documents")
        total_chunks = sum(len(chunks) for _, chunks, _ in results)
        total_embeddings = sum(len(embeddings) for _, _, embeddings in results)
        logging.info(f"Created {total_chunks} chunks")
        logging.info(f"Generated {total_embeddings} embeddings")
    
    elif args.text:
        logging.info("Processing text input")
        document, chunks, embedded_chunks = pipeline.process_text(args.text)
        
        logging.info(f"Processed document: {document.doc_id}")
        logging.info(f"Created {len(chunks)} chunks")
        logging.info(f"Generated {len(embedded_chunks)} embeddings")
    
    else:
        logging.error("No input provided. Use --file, --directory, or --text")


def retrieve_command(args: argparse.Namespace) -> None:
    """Handle the retrieve command.
    
    Args:
        args: Command line arguments
    """
    logging.info("Retrieval functionality not yet implemented")
    # This will be implemented in the future


def infer_command(args: argparse.Namespace) -> None:
    """Handle the infer command.
    
    Args:
        args: Command line arguments
    """
    logging.info("Inference functionality not yet implemented")
    # This will be implemented in the future


def main() -> None:
    """Main entry point for the Limnos CLI."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="Limnos Command Line Interface")
    
    # Global arguments
    parser.add_argument("--config", type=str, default="./config/pipeline_config.yaml",
                      help="Path to configuration file or directory")
    parser.add_argument("--log-level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Logging level")
    parser.add_argument("--storage", type=str, default="redis",
                      choices=["redis", "disk", "hybrid"],
                      help="Storage backend to use")
    parser.add_argument("--embedding-model", type=str, default="vllm",
                      choices=["vllm", "local", "openai"],
                      help="Embedding model to use")
    parser.add_argument("--use-gpu", action="store_true",
                      help="Use GPU for embedding generation if available")
    
    # Subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the system")
    ingest_input = ingest_parser.add_mutually_exclusive_group(required=True)
    ingest_input.add_argument("--file", type=str, help="Path to file to ingest")
    ingest_input.add_argument("--directory", type=str, help="Path to directory to ingest")
    ingest_input.add_argument("--text", type=str, help="Text to ingest")
    ingest_parser.add_argument("--recursive", action="store_true", help="Recursively process directory")
    
    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve information from the system")
    retrieve_parser.add_argument("--query", type=str, required=True, help="Query to search for")
    retrieve_parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    retrieve_parser.add_argument("--use-paths", action="store_true", help="Use path finding for retrieval")
    
    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Generate inferences using the system")
    infer_parser.add_argument("--query", type=str, required=True, help="Query to answer")
    infer_parser.add_argument("--stream", action="store_true", help="Stream the response")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        # Load configuration
        load_configuration(args.config)
        
        # Override configuration with command line arguments
        if args.storage:
            config.set("storage.type", args.storage)
        
        if args.embedding_model:
            config.set("models.embedding_model.type", args.embedding_model)
            
        if args.use_gpu:
            config.set("models.embedding_model.device", "cuda")
        elif "device" not in config.get("models.embedding_model", {}):
            # Only set CPU if not already specified and --use-gpu not provided
            config.set("models.embedding_model.device", "cpu")
        
        # Discover plugins
        discover_plugins()
        
        # Test storage connection
        if not test_storage_connection():
            logging.error("Failed to connect to storage backend. Please check your configuration.")
            sys.exit(1)
        
        # Handle command
        if args.command == "ingest":
            ingest_command(args)
        elif args.command == "retrieve":
            retrieve_command(args)
        elif args.command == "infer":
            infer_command(args)
        else:
            parser.print_help()
    
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
