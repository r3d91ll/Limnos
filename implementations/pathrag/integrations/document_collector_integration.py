"""
Universal Document Collector Integration for PathRAG (Refactored)

This module provides integration components to connect PathRAG with the
Universal Document Collector, handling document processing, path extraction,
and storage while maintaining the separation between universal and
framework-specific metadata. This refactored version uses Pydantic models and
the new metadata architecture.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional, Union, Callable
import shutil
import hashlib
from datetime import datetime
import traceback

# Import PathRAG components
from ..core.entity_extractor import EntityExtractor
from ..core.relationship_extractor import RelationshipExtractor
from ..core.path_constructor import PathConstructor
from ..core.extraction_pipeline import ExtractionPipeline
from ..core.path_structures import Path as PathStructure
from ..core.path_structures import PathNode, PathEdge, PathIndex
from ..core.path_vector_store import PathVectorStore
from ..core.path_storage_manager import PathStorageManager

# Import Universal Document Collector components
from limnos.ingest.collectors.universal_collector_refactored import UniversalDocumentCollector
from limnos.ingest.collectors.metadata_interface import MetadataFormat, MetadataExtensionPoint
from limnos.ingest.collectors.metadata_factory import MetadataPreprocessorFactory
from limnos.ingest.collectors.metadata_provider import MetadataProvider

# Import Pydantic models
from limnos.ingest.models.document import Document
from limnos.ingest.models.metadata import UniversalMetadata, PathRAGMetadata

# Configure logging
logger = logging.getLogger(__name__)

class DocumentProcessingError(Exception):
    """Exception raised for errors during document processing."""
    pass

class DocumentCollectorIntegration:
    """
    Integration with the Universal Document Collector for PathRAG.
    
    This component processes documents from the Universal Document Collector,
    extracts paths, and stores them in the PathRAG-specific storage while
    maintaining proper separation of universal and framework-specific metadata.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the document collector integration with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Base directory for source documents (universal storage)
        self.source_documents_dir = Path(self.config.get(
            'source_documents_dir', 
            '/home/todd/ML-Lab/Olympus/limnos/data/source_documents'
        ))
        
        # Base directory for PathRAG-specific data
        self.pathrag_data_dir = Path(self.config.get(
            'pathrag_data_dir',
            '/home/todd/ML-Lab/Olympus/limnos/data/implementations/pathrag'
        ))
        
        # Create required directories
        self.pathrag_data_dir.mkdir(parents=True, exist_ok=True)
        (self.pathrag_data_dir / "paths").mkdir(exist_ok=True)
        (self.pathrag_data_dir / "metadata").mkdir(exist_ok=True)
        (self.pathrag_data_dir / "vectors").mkdir(exist_ok=True)
        (self.pathrag_data_dir / "logs").mkdir(exist_ok=True)
        
        # Initialize the Universal Document Collector
        self.universal_collector = self._initialize_universal_collector()
        
        # Initialize PathRAG components
        self._initialize_components()
        
        # Track processed documents
        self.processed_document_ids = self._load_processed_document_ids()
    
    def _initialize_universal_collector(self) -> UniversalDocumentCollector:
        """Initialize the Universal Document Collector."""
        collector = UniversalDocumentCollector()
        collector_config = {
            'source_dir': str(self.source_documents_dir),
            'processor_type': 'general'  # Use general processor as default
        }
        collector.initialize(collector_config)
        
        # Register PathRAG metadata preprocessor
        self._register_metadata_preprocessor(collector)
        
        return collector
    
    def _register_metadata_preprocessor(self, collector: UniversalDocumentCollector) -> None:
        """
        Register the PathRAG metadata preprocessor with the Universal Document Collector.
        
        Args:
            collector: Universal Document Collector instance
        """
        # Using the factory to create the preprocessor
        pathrag_preprocessor = MetadataPreprocessorFactory.create_preprocessor(
            'pathrag',
            {'output_dir': str(self.pathrag_data_dir)}
        )
        
        # Register with the collector
        collector.register_extension_point('pathrag', pathrag_preprocessor)
        
        logger.info("PathRAG metadata preprocessor registered with the Universal Document Collector")
    
    def _initialize_components(self) -> None:
        """Initialize PathRAG components."""
        # Initialize extraction pipeline
        extraction_config = self.config.get('extraction_pipeline_config', {})
        extraction_config.update({
            'paths_dir': str(self.pathrag_data_dir / "paths"),
            'metadata_dir': str(self.pathrag_data_dir / "metadata")
        })
        self.extraction_pipeline = ExtractionPipeline(extraction_config)
        
        # Initialize storage manager
        storage_config = self.config.get('storage_manager_config', {})
        storage_config.update({
            'paths_dir': str(self.pathrag_data_dir / "paths"),
            'vectors_dir': str(self.pathrag_data_dir / "vectors")
        })
        self.storage_manager = PathStorageManager(storage_config)
    
    def _load_processed_document_ids(self) -> Set[str]:
        """
        Load the set of already processed document IDs.
        
        Returns:
            Set of processed document IDs
        """
        processed_ids_file = self.pathrag_data_dir / "processed_documents.json"
        
        if processed_ids_file.exists():
            try:
                with open(processed_ids_file, 'r') as f:
                    return set(json.load(f))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading processed document IDs: {e}")
                return set()
        else:
            return set()
    
    def _save_processed_document_ids(self) -> None:
        """Save the set of processed document IDs."""
        processed_ids_file = self.pathrag_data_dir / "processed_documents.json"
        
        try:
            with open(processed_ids_file, 'w') as f:
                json.dump(list(self.processed_document_ids), f, indent=2)
        except IOError as e:
            logger.error(f"Error saving processed document IDs: {e}")
    
    def process_document(self, doc_id: str) -> bool:
        """
        Process a document with the PathRAG pipeline.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if processing was successful, False otherwise
        """
        logger.info(f"Processing document: {doc_id}")
        
        try:
            # Get document from the Universal Document Collector
            doc_result = self.universal_collector.get_document(doc_id)
            if not doc_result:
                logger.error(f"Document not found: {doc_id}")
                return False
            
            document, file_paths = doc_result
            
            # Get PathRAG-specific metadata
            pathrag_metadata = self.universal_collector.get_framework_metadata(doc_id, 'pathrag')
            if not pathrag_metadata:
                logger.error(f"PathRAG metadata not found for document: {doc_id}")
                return False
            
            # Extract paths using the extraction pipeline
            paths = self._extract_paths(document, pathrag_metadata)
            
            # Store paths and vectors
            self._store_paths(paths, doc_id, pathrag_metadata)
            
            # Mark document as processed
            self.processed_document_ids.add(doc_id)
            self._save_processed_document_ids()
            
            # Update processing status
            extension_point = self.universal_collector._metadata_provider.get_extension_point('pathrag')
            if extension_point:
                extension_point.update_processing_status(doc_id, {
                    'metadata_processed': True,
                    'paths_extracted': True,
                    'paths_stored': True
                })
            
            logger.info(f"Document {doc_id} processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _extract_paths(self, document: Document, pathrag_metadata: Dict[str, Any]) -> List[PathStructure]:
        """
        Extract paths from a document.
        
        Args:
            document: Document
            pathrag_metadata: PathRAG-specific metadata
            
        Returns:
            List of extracted paths
        """
        logger.info(f"Extracting paths from document: {document.doc_id}")
        
        # Get path extraction configuration
        if 'path_extraction_config' in pathrag_metadata:
            config = pathrag_metadata['path_extraction_config']
        else:
            # Fallback for older metadata format
            extension_point = self.universal_collector._metadata_provider.get_extension_point('pathrag')
            if extension_point:
                config = extension_point.get_path_extraction_config(document.doc_id)
            else:
                config = {}
        
        # Extract paths
        paths = self.extraction_pipeline.extract_paths(
            document_id=document.doc_id,
            document_content=document.content,
            metadata=pathrag_metadata,
            config=config
        )
        
        # Save paths to file
        paths_path = self.pathrag_data_dir / "paths" / f"{document.doc_id}.json"
        paths_serializable = [path.to_dict() for path in paths]
        
        with open(paths_path, 'w') as f:
            json.dump(paths_serializable, f, indent=2)
        
        logger.info(f"Extracted {len(paths)} paths from document {document.doc_id}")
        return paths
    
    def _store_paths(self, paths: List[PathStructure], doc_id: str, pathrag_metadata: Dict[str, Any]) -> None:
        """
        Store paths and generate vectors.
        
        Args:
            paths: List of paths
            doc_id: Document ID
            pathrag_metadata: PathRAG-specific metadata
        """
        logger.info(f"Storing paths for document: {doc_id}")
        
        # Get storage configuration
        if 'storage_config' in pathrag_metadata:
            config = pathrag_metadata['storage_config']
        else:
            config = {}
        
        # Store paths and their vectors
        self.storage_manager.store_paths(
            document_id=doc_id,
            paths=paths,
            config=config
        )
        
        logger.info(f"Stored {len(paths)} paths for document {doc_id}")
    
    def process_all_documents(self) -> Tuple[int, int]:
        """
        Process all documents that haven't been processed yet.
        
        Returns:
            Tuple of (number of documents processed, number of documents with errors)
        """
        # Get all document IDs from the Universal Document Collector
        all_doc_ids = set(self.universal_collector.list_documents())
        
        # Filter out already processed documents
        unprocessed_doc_ids = all_doc_ids - self.processed_document_ids
        
        logger.info(f"Found {len(unprocessed_doc_ids)} unprocessed documents")
        
        success_count = 0
        error_count = 0
        
        for doc_id in unprocessed_doc_ids:
            if self.process_document(doc_id):
                success_count += 1
            else:
                error_count += 1
        
        return success_count, error_count
    
    def get_document_processing_status(self, doc_id: str) -> Dict[str, Any]:
        """
        Get the processing status of a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Status dictionary
        """
        status = {
            'document_id': doc_id,
            'exists': False,
            'processed': False,
            'paths_extracted': False,
            'paths_stored': False,
            'path_count': 0
        }
        
        # Check if document exists
        if not self.universal_collector.document_exists(doc_id):
            return status
        
        status['exists'] = True
        
        # Check if document has been processed
        status['processed'] = doc_id in self.processed_document_ids
        
        # Check for paths
        paths_file = self.pathrag_data_dir / "paths" / f"{doc_id}.json"
        if paths_file.exists():
            status['paths_extracted'] = True
            try:
                with open(paths_file, 'r') as f:
                    paths_data = json.load(f)
                    status['path_count'] = len(paths_data)
            except:
                pass
        
        # Check for vectors
        vectors_file = self.pathrag_data_dir / "vectors" / f"{doc_id}.json"
        if vectors_file.exists():
            status['paths_stored'] = True
        
        return status
    
    def reprocess_document(self, doc_id: str, force: bool = False) -> bool:
        """
        Reprocess a document that has already been processed.
        
        Args:
            doc_id: Document ID
            force: If True, reprocess even if document is already in processed list
            
        Returns:
            True if reprocessing was successful, False otherwise
        """
        # Check if document has been processed already
        if doc_id in self.processed_document_ids and not force:
            logger.info(f"Document {doc_id} has already been processed. Use force=True to reprocess.")
            return False
        
        # Remove existing paths and vectors
        self._cleanup_document_artifacts(doc_id)
        
        # Remove from processed list
        if doc_id in self.processed_document_ids:
            self.processed_document_ids.remove(doc_id)
            self._save_processed_document_ids()
        
        # Process document
        return self.process_document(doc_id)
    
    def _cleanup_document_artifacts(self, doc_id: str) -> None:
        """
        Remove all artifacts for a document.
        
        Args:
            doc_id: Document ID
        """
        # Remove paths file
        paths_file = self.pathrag_data_dir / "paths" / f"{doc_id}.json"
        if paths_file.exists():
            paths_file.unlink()
        
        # Remove vectors file
        vectors_file = self.pathrag_data_dir / "vectors" / f"{doc_id}.json"
        if vectors_file.exists():
            vectors_file.unlink()
        
        # Remove metadata file
        metadata_file = self.pathrag_data_dir / "metadata" / f"{doc_id}.json"
        if metadata_file.exists():
            metadata_file.unlink()
            
        logger.info(f"Cleaned up artifacts for document {doc_id}")
    
    def get_document_paths(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get all paths for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of paths as dictionaries
        """
        paths_file = self.pathrag_data_dir / "paths" / f"{doc_id}.json"
        if not paths_file.exists():
            logger.warning(f"No paths found for document {doc_id}")
            return []
        
        try:
            with open(paths_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading paths for document {doc_id}: {e}")
            return []
    
    def query_paths(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Query paths using semantic similarity.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of path results
        """
        return self.storage_manager.query_paths(query, top_k)
    
    def get_paths_by_metadata(self, metadata_filter: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get paths that match the metadata filter.
        
        Args:
            metadata_filter: Dictionary of metadata key-value pairs to filter on
            limit: Maximum number of paths to return
            
        Returns:
            List of path results
        """
        return self.storage_manager.get_paths_by_metadata(metadata_filter, limit)
    
    def get_path_by_id(self, path_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific path by ID.
        
        Args:
            path_id: Path ID
            
        Returns:
            Path if found, None otherwise
        """
        return self.storage_manager.get_path_by_id(path_id)
    
    def batch_process_documents(self, batch_size: int = 10) -> Tuple[int, int]:
        """
        Process documents in batches.
        
        Args:
            batch_size: Number of documents to process in each batch
            
        Returns:
            Tuple of (total documents processed, documents with errors)
        """
        # Get all document IDs from the Universal Document Collector
        all_doc_ids = set(self.universal_collector.list_documents())
        
        # Filter out already processed documents
        unprocessed_doc_ids = list(all_doc_ids - self.processed_document_ids)
        
        logger.info(f"Found {len(unprocessed_doc_ids)} unprocessed documents")
        
        total_processed = 0
        total_errors = 0
        
        # Process in batches
        for i in range(0, len(unprocessed_doc_ids), batch_size):
            batch = unprocessed_doc_ids[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(unprocessed_doc_ids)-1)//batch_size + 1} with {len(batch)} documents")
            
            success_count = 0
            error_count = 0
            
            for doc_id in batch:
                if self.process_document(doc_id):
                    success_count += 1
                else:
                    error_count += 1
                    
            total_processed += success_count
            total_errors += error_count
            
            logger.info(f"Batch complete: {success_count} succeeded, {error_count} failed")
        
        return total_processed, total_errors
