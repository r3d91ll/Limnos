#!/usr/bin/env python3
"""
Reset RAG Database

This script helps transition from the old PathRAG database to the new RAG Dataset Builder.
It creates the necessary directory structure and can optionally delete previous datasets.

Usage:
    python reset_database.py --config ../config/config.yaml [--delete-old] [--preserve-source-docs]
"""

import os
import sys
import argparse
import shutil
import logging
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("reset_database")

def load_config(config_path):
    """Load configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        sys.exit(1)

def create_directory_structure(config):
    """Create the necessary directory structure."""
    # Create source documents directory
    source_dir = os.path.abspath(os.path.expanduser(config.get('source_documents_dir', '../source_documents')))
    os.makedirs(source_dir, exist_ok=True)
    logger.info(f"Created source documents directory: {source_dir}")
    
    # Create necessary subdirectories
    for subdir in ['papers', 'docs', 'code']:
        os.makedirs(os.path.join(source_dir, subdir), exist_ok=True)
    
    # Create output directory
    output_dir = os.path.abspath(os.path.expanduser(config.get('output_dir', '../rag_databases/current')))
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    # Create output subdirectories based on formatter type
    formatter_type = config.get('output', {}).get('format', 'pathrag')
    if formatter_type == 'pathrag':
        for subdir in ['chunks', 'metadata', 'embeddings', 'graph']:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    elif formatter_type == 'vector_db':
        for subdir in ['texts', 'metadata', 'vectors']:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # Create logs directory
    logs_dir = os.path.abspath('./logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    return source_dir, output_dir

def delete_old_database(output_dir, preserve_source=False):
    """Delete old database files while optionally preserving source documents."""
    try:
        if os.path.exists(output_dir):
            logger.info(f"Deleting old database at: {output_dir}")
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            logger.info("Old database deleted successfully")
    except Exception as e:
        logger.error(f"Error deleting old database: {e}")

def main():
    parser = argparse.ArgumentParser(description="Reset RAG Database")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--delete-old", action="store_true", help="Delete old database")
    parser.add_argument("--preserve-source-docs", action="store_true", help="Preserve source documents when deleting old database")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directory structure
    source_dir, output_dir = create_directory_structure(config)
    
    # Delete old database if requested
    if args.delete_old:
        delete_old_database(output_dir, args.preserve_source_docs)
    
    logger.info("Database reset complete. Ready for new dataset building.")
    logger.info(f"Source documents will be stored in: {source_dir}")
    logger.info(f"Processed database will be stored in: {output_dir}")

if __name__ == "__main__":
    main()
