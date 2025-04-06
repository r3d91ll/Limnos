"""
Universal Document Collector for Limnos (Refactored).

This module provides a document collector that handles the entire workflow
from receiving a document to processing it and storing it with universal metadata.
This refactored version uses Pydantic models and simplified inheritance.
"""

import os
import logging
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Set
import uuid
from datetime import datetime

from limnos.ingest.models.document import Document
from limnos.ingest.models.metadata import UniversalMetadata, DocumentType, validate_metadata
from limnos.ingest.collectors.collector_interface import DocumentCollector
from limnos.ingest.collectors.metadata_provider import MetadataProvider
from limnos.ingest.collectors.metadata_interface import MetadataFormat, MetadataExtensionPoint


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
            from limnos.ingest.processors.academic_processor import AcademicPaperProcessor
            self._document_processor = AcademicPaperProcessor()
        else:
            from limnos.ingest.processors.basic_processor import BasicDocumentProcessor
            self._document_processor = BasicDocumentProcessor()
        
        # Configure the processor
        processor_config = config.get('processor_config', {})
        self._document_processor.initialize(processor_config)
        
        # Initialize the metadata provider
        self._metadata_provider = MetadataProvider(self._source_dir)
        
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
        processed_doc = self._document_processor.process_file(file_path)
        
        # Convert to our new Document model
        document = Document(
            doc_id=processed_doc.metadata.get('doc_id', str(uuid.uuid4())),
            content=processed_doc.content,
            metadata=processed_doc.metadata,
            file_path=file_path
        )
        
        # Create storage paths
        original_filename = file_path.name
        metadata_filename = f"{document.doc_id}.json"
        
        # Create document directory using doc_id
        doc_dir = self._source_dir / document.doc_id
        os.makedirs(doc_dir, exist_ok=True)
        
        # Store original file
        original_path = doc_dir / original_filename
        shutil.copy2(file_path, original_path)
        
        # Ensure we have a UniversalMetadata instance
        if not isinstance(document.metadata, UniversalMetadata):
            # Convert or update metadata dictionary
            metadata_dict = document.metadata if isinstance(document.metadata, dict) else {}
            metadata_dict.update({
                'doc_id': document.doc_id,
                'storage_path': str(original_path),
                'metadata_path': str(doc_dir / metadata_filename),
                'collection_timestamp': datetime.now(),
                'filename': original_filename,
                'extension': file_path.suffix,
                'size_bytes': os.path.getsize(file_path)
            })
            
            # Create UniversalMetadata instance
            universal_metadata = UniversalMetadata.from_dict(metadata_dict)
            document.metadata = universal_metadata
        else:
            # Update existing UniversalMetadata
            document.metadata.storage_path = str(original_path)
            document.metadata.metadata_path = str(doc_dir / metadata_filename)
            document.metadata.collection_timestamp = datetime.now()
            document.metadata.filename = original_filename
            document.metadata.extension = file_path.suffix
            document.metadata.size_bytes = os.path.getsize(file_path)
        
        # Store metadata using MetadataProvider
        metadata_path = self._metadata_provider.save_metadata(document.doc_id, document.metadata)
        
        # Update document index
        self._update_document_index(document.doc_id, document.metadata.to_dict())
        
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
                files.extend(directory_path.glob(f"**/*{ext}"))
        else:
            for ext in file_extensions:
                files.extend(directory_path.glob(f"*{ext}"))
        
        # Process each file
        results = []
        for file_path in files:
            try:
                result = self.collect_file(file_path)
                results.append(result)
            except Exception as e:
                self._logger.error(f"Error processing file {file_path}: {e}")
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Tuple[Document, Dict[str, Path]]]:
        """Retrieve a previously collected document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Tuple of (document, file_paths) if found, None otherwise
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
        
        # Check if document exists
        doc_dir = self._source_dir / doc_id
        if not doc_dir.exists():
            return None
        
        # Get metadata
        metadata = self._metadata_provider.get_metadata(doc_id, MetadataFormat.PYDANTIC)
        if not metadata:
            return None
        
        # Build file_paths dict
        file_paths = {}
        if metadata.storage_path:
            file_paths['original'] = Path(metadata.storage_path)
        if metadata.metadata_path:
            file_paths['metadata'] = Path(metadata.metadata_path)
        
        # Load original file content
        content = ""
        if 'original' in file_paths and file_paths['original'].exists():
            try:
                # For text files, read directly
                if file_paths['original'].suffix.lower() in ['.txt', '.md']:
                    with open(file_paths['original'], 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    # For other file types, reprocess if needed
                    processed_doc = self._document_processor.process_file(file_paths['original'])
                    content = processed_doc.content
            except Exception as e:
                self._logger.error(f"Error loading content for {doc_id}: {e}")
        
        # Create document instance
        document = Document(
            doc_id=doc_id,
            content=content,
            metadata=metadata,
            file_path=file_paths.get('original', None)
        )
        
        return document, file_paths
    
    def list_documents(self) -> List[str]:
        """List all collected documents.
        
        Returns:
            List of document IDs
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
        
        # Return document IDs from index
        return list(self._document_index.keys())
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a collected document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if deleted, False otherwise
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
        
        # Check if document exists
        doc_dir = self._source_dir / doc_id
        if not doc_dir.exists():
            return False
        
        # Delete directory and all contents
        try:
            shutil.rmtree(doc_dir)
            
            # Remove from index
            if doc_id in self._document_index:
                del self._document_index[doc_id]
                
            return True
        except Exception as e:
            self._logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def _build_document_index(self) -> None:
        """Build an index of collected documents."""
        self._document_index = {}
        
        # Get all document IDs from metadata provider
        doc_ids = self._metadata_provider.get_document_ids()
        
        # Build index
        for doc_id in doc_ids:
            metadata = self._metadata_provider.get_metadata(doc_id)
            if metadata:
                self._document_index[doc_id] = metadata
        
        self._last_indexed = datetime.now()
    
    def _update_document_index(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Update the document index with a new or updated document."""
        self._document_index[doc_id] = metadata
    
    def register_extension_point(self, framework_name: str, extension: MetadataExtensionPoint) -> None:
        """Register an extension point for framework-specific metadata processing.
        
        Args:
            framework_name: Name of the framework
            extension: Extension point implementation
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
        
        # Register with metadata provider
        self._metadata_provider.register_extension_point(framework_name, extension)
        
        # Add to registered frameworks
        self._registered_frameworks.add(framework_name)
    
    def get_framework_metadata(self, doc_id: str, framework: str) -> Optional[Dict[str, Any]]:
        """Get framework-specific metadata for a document.
        
        Args:
            doc_id: Document ID
            framework: Framework name
            
        Returns:
            Framework-specific metadata if found, None otherwise
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
        
        # Check if framework is registered
        if framework not in self._registered_frameworks:
            raise ValueError(f"Framework not registered: {framework}")
        
        try:
            # Process universal metadata with framework extension
            return self._metadata_provider.process_metadata_with_extension(doc_id, framework)
        except ValueError:
            return None
    
    def get_processing_status(self, doc_id: str, framework: Optional[str] = None) -> Dict[str, Any]:
        """Get the processing status for a document.
        
        Args:
            doc_id: Document ID
            framework: Optional framework name to get framework-specific status
            
        Returns:
            Processing status dictionary
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
        
        # Check if document exists
        if doc_id not in self._document_index:
            return {'exists': False}
        
        status = {'exists': True}
        
        # If framework specified, get framework-specific status
        if framework:
            if framework not in self._registered_frameworks:
                return {'exists': True, 'framework_registered': False}
            
            extension = self._metadata_provider.get_extension_point(framework)
            if extension:
                framework_status = extension.get_processing_status(doc_id)
                status.update({'framework_registered': True, framework: framework_status})
            else:
                status.update({'framework_registered': False})
        
        return status
    
    def get_registered_frameworks(self) -> Set[str]:
        """Get the set of registered frameworks.
        
        Returns:
            Set of framework names
        """
        return self._registered_frameworks.copy()
    
    def process_document_for_framework(self, doc_id: str, framework: str) -> Optional[Dict[str, Any]]:
        """Process a document for a specific framework.
        
        This method:
        1. Gets the universal metadata for the document
        2. Processes it with the framework's extension point
        3. Returns the framework-specific metadata
        
        The framework is responsible for storing any derived data.
        
        Args:
            doc_id: Document ID
            framework: Framework name
            
        Returns:
            Framework-specific metadata if processing successful, None otherwise
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
        
        # Check if framework is registered
        if framework not in self._registered_frameworks:
            raise ValueError(f"Framework not registered: {framework}")
        
        # Get universal metadata
        metadata = self._metadata_provider.get_metadata(doc_id)
        if not metadata:
            return None
        
        # Get extension point
        extension = self._metadata_provider.get_extension_point(framework)
        if not extension:
            return None
        
        # Process metadata
        framework_metadata = extension.transform_metadata(metadata)
        
        # Update processing status
        extension.update_processing_status(doc_id, {'metadata_processed': True})
        
        return framework_metadata
    
    def bulk_process_for_framework(self, doc_ids: List[str], framework: str) -> Dict[str, Dict[str, Any]]:
        """Process multiple documents for a specific framework.
        
        Args:
            doc_ids: List of document IDs
            framework: Framework name
            
        Returns:
            Dictionary mapping document IDs to framework-specific metadata
        """
        if not self._initialized:
            raise RuntimeError("Collector not initialized")
        
        # Check if framework is registered
        if framework not in self._registered_frameworks:
            raise ValueError(f"Framework not registered: {framework}")
        
        # Process in bulk
        return self._metadata_provider.bulk_process_metadata(doc_ids, framework)
