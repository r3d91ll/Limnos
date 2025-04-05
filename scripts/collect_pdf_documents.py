#!/usr/bin/env python
"""
PDF Document Collection Script for Limnos.

This script provides a command-line interface for the Universal Document Collector,
allowing users to collect PDF documents and store them with universal metadata.

Usage:
    python collect_pdf_documents.py collect-file --file_path /path/to/document.pdf
    python collect_pdf_documents.py collect-dir --dir_path /path/to/directory --recursive
    python collect_pdf_documents.py list
    python collect_pdf_documents.py get --doc_id <document_id>
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from limnos.ingest.collectors.pdf.document_collector import UniversalDocumentCollector


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def collect_file(args):
    """Collect a single file."""
    collector = UniversalDocumentCollector(args.source_dir)
    
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return
    
    print(f"Collecting file: {file_path}")
    document, paths = collector.collect_file(file_path)
    
    print(f"\nDocument collected successfully!")
    print(f"Document ID: {document['doc_id']}")
    print(f"Title: {document['metadata'].get('title', 'Unknown')}")
    print(f"Authors: {', '.join(document['metadata'].get('authors', ['Unknown']))}")
    print(f"\nStored at:")
    print(f"  Original: {paths['original']}")
    print(f"  Metadata: {paths['metadata']}")
    
    # Print some of the metadata
    print("\nMetadata preview:")
    metadata_preview = {k: v for k, v in document['metadata'].items() 
                      if k in ['title', 'authors', 'abstract', 'doc_type', 'file_size']}
    print(json.dumps(metadata_preview, indent=2))


def collect_directory(args):
    """Collect all files in a directory."""
    collector = UniversalDocumentCollector(args.source_dir)
    
    dir_path = Path(args.dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: Directory {dir_path} does not exist")
        return
    
    print(f"Collecting files from directory: {dir_path}")
    print(f"Recursive: {args.recursive}")
    
    # File extensions to collect
    extensions = args.extensions.split(',') if args.extensions else ['.pdf']
    print(f"File extensions: {extensions}")
    
    results = collector.collect_directory(dir_path, args.recursive, extensions)
    
    print(f"\nCollected {len(results)} documents:")
    for document, paths in results:
        print(f"  - {document['metadata'].get('title', 'Unknown')} (ID: {document['doc_id']})")


def list_documents(args):
    """List all collected documents."""
    collector = UniversalDocumentCollector(args.source_dir)
    
    doc_ids = collector.list_documents()
    
    print(f"Found {len(doc_ids)} documents:")
    for doc_id in doc_ids:
        result = collector.get_document(doc_id)
        if result:
            document, _ = result
            print(f"  - {document['metadata'].get('title', 'Unknown')} (ID: {doc_id})")
        else:
            print(f"  - {doc_id} (metadata unavailable)")


def get_document(args):
    """Get details of a specific document."""
    collector = UniversalDocumentCollector(args.source_dir)
    
    result = collector.get_document(args.doc_id)
    if not result:
        print(f"Error: Document {args.doc_id} not found")
        return
    
    document, paths = result
    
    print(f"Document ID: {document['doc_id']}")
    print(f"Title: {document['metadata'].get('title', 'Unknown')}")
    print(f"Authors: {', '.join(document['metadata'].get('authors', ['Unknown']))}")
    print(f"\nStored at:")
    print(f"  Original: {paths['original']}")
    print(f"  Metadata: {paths['metadata']}")
    
    # Print all metadata
    print("\nFull Metadata:")
    print(json.dumps(document['metadata'], indent=2))


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Limnos PDF Document Collection Tool')
    parser.add_argument('--source_dir', help='Custom source documents directory')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Collect file command
    collect_file_parser = subparsers.add_parser('collect-file', help='Collect a single file')
    collect_file_parser.add_argument('--file_path', required=True, help='Path to the file to collect')
    collect_file_parser.set_defaults(func=collect_file)
    
    # Collect directory command
    collect_dir_parser = subparsers.add_parser('collect-dir', help='Collect all files in a directory')
    collect_dir_parser.add_argument('--dir_path', required=True, help='Path to the directory to collect')
    collect_dir_parser.add_argument('--recursive', action='store_true', help='Collect files recursively')
    collect_dir_parser.add_argument('--extensions', help='Comma-separated list of file extensions to collect')
    collect_dir_parser.set_defaults(func=collect_directory)
    
    # List documents command
    list_parser = subparsers.add_parser('list', help='List all collected documents')
    list_parser.set_defaults(func=list_documents)
    
    # Get document command
    get_parser = subparsers.add_parser('get', help='Get details of a specific document')
    get_parser.add_argument('--doc_id', required=True, help='Document ID')
    get_parser.set_defaults(func=get_document)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging()
    args.func(args)


if __name__ == '__main__':
    main()
