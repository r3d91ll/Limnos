"""
Universal Document Collector for Limnos.

This module provides a document collector that handles the entire workflow
from receiving a document to processing it and storing it with universal metadata.
"""

import os
import logging
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import uuid

from limnos.ingest.interface import Document
from limnos.ingest.processors.academic_processor import AcademicPaperProcessor
from limnos.ingest.collectors.metadata_schema import UniversalMetadataSchema, DocumentType, validate_metadata
from limnos.ingest.collectors.collector_interface import DocumentCollector


class UniversalDocumentCollector(DocumentCollector):
    """Universal Document Collector for the Limnos RAG system.
    
    This collector handles:
    1. Receiving documents from different sources
    2. Processing them to extract content and metadata
    3. Storing the original documents and metadata in the source_documents directory
    """
    
    def __init__(self):
        """Initialize the universal document collector."""
        self._logger = logging.getLogger(__name__)
        self._config = {}
        self._initialized = False
        self._document_processor = None
        self._source_dir = None
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the collector with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config
        
        # Set up the source documents directory
        self._source_dir = Path(config.get('source_dir', '/home/todd/ML-Lab/Olympus/limnos/data/source_documents'))
        os.makedirs(self._source_dir, exist_ok=True)
        
        # Initialize the document processor
        processor_type = config.get('processor_type', 'academic')
        if processor_type == 'academic':
            self._document_processor = AcademicPaperProcessor()
        else:
            from limnos.ingest.processors.basic_processor import BasicDocumentProcessor
            self._document_processor = BasicDocumentProcessor()
        
        # Configure the processor
        processor_config = config.get('processor_config', {})
        self._document_processor.initialize(processor_config)
        
        self._initialized = True
        self._logger.info(f"Initialized universal document collector with source dir: {self._source_dir}")
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "document_collector"
    
    @property
    def component_name(self) -> str:
        """Return the name of this component."""
        return "universal_collector"
        
    def get_default_config(self) -> Dict[str, Any]:
        """Return the default configuration for this component.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'source_dir': '/home/todd/ML-Lab/Olympus/limnos/data/source_documents',
            'processor_type': 'academic',
            'processor_config': {
                'custom_extensions': {}
            }
        }
    
    def collect_file(self, file_path: Path) -> Tuple[Document, Dict[str, Path]]:
        """Collect a single file, process it, and store with metadata.
        
        Args:
            file_path: Path to the file to collect
            
        Returns:
            Tuple of (document, file_paths) where file_paths is a dict with paths to:
            - original: Path to the stored original document
            - metadata: Path to the stored metadata JSON file
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
        
        # Convert to Path object
        file_path = Path(file_path)
        
        # Process the document
        document = self._document_processor.process_file(file_path)
        
        # Generate unique ID for storage if not already set
        doc_id = document.metadata.get('doc_id', str(uuid.uuid4()))
        document.metadata['doc_id'] = doc_id
        
        # Create storage paths
        original_filename = file_path.name
        metadata_filename = f"{file_path.stem}.json"
        
        # Create document directory using doc_id
        doc_dir = self._source_dir / doc_id
        os.makedirs(doc_dir, exist_ok=True)
        
        # Store original file
        original_path = doc_dir / original_filename
        shutil.copy2(file_path, original_path)
        
        # Add storage path to metadata
        document.metadata['storage_path'] = str(original_path)
        document.metadata['metadata_path'] = str(doc_dir / metadata_filename)
        
        # Validate and store metadata according to schema
        errors = validate_metadata(document.metadata)
        if errors:
            self._logger.warning(f"Metadata validation warnings for {doc_id}: {', '.join(errors)}")
            
        # Ensure metadata conforms to schema
        schema = UniversalMetadataSchema.from_dict(document.metadata)
        standardized_metadata = schema.to_dict()
        
        # Update document with standardized metadata
        document.metadata = standardized_metadata
        
        # Store metadata
        metadata_path = doc_dir / metadata_filename
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(standardized_metadata, f, indent=2)
        
        return document, {
            'original': original_path,
            'metadata': metadata_path
        }
    
    def collect_directory(self, directory_path: Path, recursive: bool = True, 
                          file_extensions: Optional[List[str]] = None) -> List[Tuple[Document, Dict[str, Path]]]:
        """Collect all files in a directory, process them, and store with metadata.
        
        Args:
            directory_path: Path to the directory to collect
            recursive: Whether to process subdirectories
            file_extensions: Optional list of file extensions to process
            
        Returns:
            List of tuples containing (document, file_paths) for each collected document
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
        
        directory_path = Path(directory_path)
        
        # Default to PDF files if no extensions specified
        if not file_extensions:
            file_extensions = ['.pdf']
        
        # Get all files
        files = []
        if recursive:
            for ext in file_extensions:
                files.extend(directory_path.glob(f'**/*{ext}'))
        else:
            for ext in file_extensions:
                files.extend(directory_path.glob(f'*{ext}'))
        
        # Collect each file
        results = []
        for file_path in files:
            try:
                result = self.collect_file(file_path)
                results.append(result)
                self._logger.info(f"Collected {file_path}")
            except Exception as e:
                self._logger.error(f"Error collecting {file_path}: {e}")
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Tuple[Document, Dict[str, Path]]]:
        """Retrieve a previously collected document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Tuple of (document, file_paths) if found, None otherwise
        """
        doc_dir = self._source_dir / doc_id
        
        if not doc_dir.exists():
            return None
        
        # Find metadata file
        metadata_files = list(doc_dir.glob('*.json'))
        if not metadata_files:
            return None
        
        metadata_path = metadata_files[0]
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Find original file
        original_files = [f for f in doc_dir.iterdir() if f.is_file() and f.suffix != '.json']
        if not original_files:
            return None
        
        original_path = original_files[0]
        
        # Create document
        document = Document(doc_id=doc_id, content="", metadata=metadata)
        
        # Load content if it's text-based
        if original_path.suffix.lower() in ['.txt', '.md']:
            with open(original_path, 'r', encoding='utf-8') as f:
                document.content = f.read()
        
        return document, {
            'original': original_path,
            'metadata': metadata_path
        }
    
    def list_documents(self) -> List[str]:
        """List all collected documents.
        
        Returns:
            List of document IDs
        """
        if not self._source_dir.exists():
            return []
        
        return [d.name for d in self._source_dir.iterdir() if d.is_dir()]
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a collected document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if deleted, False otherwise
        """
        doc_dir = self._source_dir / doc_id
        
        if not doc_dir.exists():
            return False
        
        # Delete directory and all contents
        shutil.rmtree(doc_dir)
        
        return True
