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
from typing import Any, Dict, List, Optional, Union, Tuple, Set
import uuid
from datetime import datetime

from limnos.ingest.interface import Document
from limnos.ingest.processors.academic_processor import AcademicPaperProcessor
from limnos.ingest.collectors.metadata_schema import UniversalMetadataSchema, DocumentType, validate_metadata
from limnos.ingest.collectors.collector_interface import DocumentCollector
from limnos.ingest.collectors.metadata_interface import (
    MetadataProvider, 
    UniversalMetadataProvider, 
    MetadataFormat, 
    MetadataExtensionPoint
)


class UniversalDocumentCollector(DocumentCollector):
    """Universal Document Collector for the Limnos RAG system.
    
    This collector handles:
    1. Receiving documents from different sources
    2. Processing them to extract content and metadata
    3. Storing the original documents and metadata in the source_documents directory
    4. Providing standardized metadata access through the MetadataProvider interface
    5. Supporting extension points for framework-specific metadata preprocessing
    """
    
    def __init__(self):
        """Initialize the universal document collector."""
        self._logger = logging.getLogger(__name__)
        self._config = {}
        self._initialized = False
        self._document_processor = None
        self._source_dir = None
        self._metadata_provider = None
        self._registered_frameworks = set()
        self._document_index = {}
        self._last_indexed = None
    
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
        
        # Initialize the metadata provider
        self._metadata_provider = UniversalMetadataProvider(self._source_dir)
        
        # Build document index
        self._build_document_index()
        
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
        metadata_filename = f"{doc_id}.json"  # Use doc_id for metadata filename for consistency
        
        # Create document directory using doc_id
        doc_dir = self._source_dir / doc_id
        os.makedirs(doc_dir, exist_ok=True)
        
        # Store original file
        original_path = doc_dir / original_filename
        shutil.copy2(file_path, original_path)
        
        # Add storage path to metadata
        document.metadata['storage_path'] = str(original_path)
        document.metadata['metadata_path'] = str(doc_dir / metadata_filename)
        document.metadata['collection_timestamp'] = datetime.now().isoformat()
        document.metadata['filename'] = original_filename
        document.metadata['extension'] = file_path.suffix
        document.metadata['size_bytes'] = os.path.getsize(file_path)
        
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
        
        # Update document index
        self._update_document_index(doc_id, standardized_metadata)
        
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
    
    def _build_document_index(self) -> None:
        """Build an index of all documents in the source directory."""
        self._document_index = {}
        
        # Iterate through all subdirectories in the source directory
        for doc_dir in self._source_dir.iterdir():
            if not doc_dir.is_dir():
                continue
                
            doc_id = doc_dir.name
            metadata_path = doc_dir / f"{doc_id}.json"
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    self._document_index[doc_id] = {
                        'metadata_path': str(metadata_path),
                        'title': metadata.get('title', 'Untitled'),
                        'doc_type': metadata.get('doc_type', 'other'),
                        'collection_timestamp': metadata.get('collection_timestamp'),
                        'size_bytes': metadata.get('size_bytes', 0)
                    }
                except Exception as e:
                    self._logger.error(f"Error loading metadata for {doc_id}: {e}")
        
        self._last_indexed = datetime.now().isoformat()
        self._logger.info(f"Built document index with {len(self._document_index)} documents")
    
    def _update_document_index(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Update the document index with a new or updated document.
        
        Args:
            doc_id: Document ID
            metadata: Document metadata
        """
        metadata_path = self._source_dir / doc_id / f"{doc_id}.json"
        self._document_index[doc_id] = {
            'metadata_path': str(metadata_path),
            'title': metadata.get('title', 'Untitled'),
            'doc_type': metadata.get('doc_type', 'other'),
            'collection_timestamp': metadata.get('collection_timestamp'),
            'size_bytes': metadata.get('size_bytes', 0)
        }
    
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
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
            
        # Return document IDs from the index for better performance
        return list(self._document_index.keys())
    
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
        
        # Remove from document index
        if doc_id in self._document_index:
            del self._document_index[doc_id]
        
        return True
        
    # Metadata Provider Interface Methods
    
    def get_metadata_provider(self) -> MetadataProvider:
        """Get the metadata provider for this collector.
        
        Returns:
            Metadata provider instance
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
            
        return self._metadata_provider
    
    def register_framework(self, framework_name: str) -> None:
        """Register a RAG framework with the collector.
        
        Args:
            framework_name: Name of the framework (e.g., 'pathrag', 'graphrag')
        """
        self._registered_frameworks.add(framework_name)
        self._logger.info(f"Registered framework: {framework_name}")
    
    def get_registered_frameworks(self) -> Set[str]:
        """Get the set of registered frameworks.
        
        Returns:
            Set of framework names
        """
        return self._registered_frameworks
    
    def register_metadata_extension_point(self, name: str, extension: MetadataExtensionPoint) -> None:
        """Register a metadata extension point.
        
        Args:
            name: Name of the extension point (typically framework name)
            extension: Extension point implementation
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
            
        self._metadata_provider.register_extension_point(name, extension)
        self._logger.info(f"Registered metadata extension point: {name}")
    
    def process_metadata_with_extension(self, document_id: str, extension_name: str) -> Dict[str, Any]:
        """Process metadata with a specific extension point.
        
        Args:
            document_id: ID of the document
            extension_name: Name of the extension point to use
            
        Returns:
            Processed metadata
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
            
        return self._metadata_provider.process_metadata_with_extension(document_id, extension_name)
    
    def get_document_index(self) -> Dict[str, Dict[str, Any]]:
        """Get the document index.
        
        Returns:
            Dictionary mapping document IDs to basic metadata
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
            
        return self._document_index.copy()
    
    def refresh_document_index(self) -> None:
        """Refresh the document index."""
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
            
        self._build_document_index()
        
    def get_metadata_for_framework(self, document_id: str, framework_name: str) -> Dict[str, Any]:
        """Get framework-specific metadata for a document.
        
        Args:
            document_id: ID of the document
            framework_name: Name of the framework
            
        Returns:
            Framework-specific metadata
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
            
        # Check if framework is registered
        if framework_name not in self._registered_frameworks:
            raise ValueError(f"Framework not registered: {framework_name}")
            
        # Check if extension point exists
        extension = self._metadata_provider.get_extension_point(framework_name)
        if not extension:
            raise ValueError(f"No extension point registered for framework: {framework_name}")
            
        # Process metadata with extension
        return self.process_metadata_with_extension(document_id, framework_name)
        
    def bulk_process_metadata(self, document_ids: List[str], framework_name: str) -> Dict[str, Dict[str, Any]]:
        """Process metadata for multiple documents with a specific framework.
        
        Args:
            document_ids: List of document IDs
            framework_name: Name of the framework
            
        Returns:
            Dictionary mapping document IDs to processed metadata
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
            
        # Check if framework is registered
        if framework_name not in self._registered_frameworks:
            raise ValueError(f"Framework not registered: {framework_name}")
            
        # Process metadata in bulk
        return self._metadata_provider.bulk_process_metadata(document_ids, framework_name)
