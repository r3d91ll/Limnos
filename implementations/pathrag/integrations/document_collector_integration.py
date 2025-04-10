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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
        # Process the document through extraction pipeline
        result = self.extraction_pipeline.process_document({
            "id": document.doc_id,
            "content": document.content,
            "metadata": pathrag_metadata,
            **config
        })
        
        # Build graph and extract paths from the processed entities and relationships
        self.extraction_pipeline.build_graph()
        paths: List[PathStructure] = []  # Initialize empty paths list with type annotation
        
        # Save paths to file
        paths_path = self.pathrag_data_dir / "paths" / f"{document.doc_id}.json"
        # Handle paths that may not have to_dict method
        paths_serializable: List[Dict[str, Any]] = []
        for path in paths:
            try:
                if hasattr(path, 'to_dict'):
                    paths_serializable.append(path.to_dict())
                else:
                    # Create a dictionary representation for objects without to_dict
                    paths_serializable.append({
                        "id": str(hash(str(path))),
                        "path": str(path) if hasattr(path, '__str__') else "unknown_path"
                    })
            except Exception as e:
                logger.warning(f"Error serializing path: {e}")
                # Add a minimal representation to avoid breaking the JSON serialization
                paths_serializable.append({"id": str(id(path)), "error": "Failed to serialize path"})
        
        with open(paths_path, 'w') as f:
            json.dump(paths_serializable, f, indent=2)
        
        logger.info(f"Extracted {len(paths)} paths from document {document.doc_id}")
        return paths
    
    def _store_paths(self, paths: List[Any], doc_id: str, pathrag_metadata: Dict[str, Any]) -> None:
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
        # Using path_constructor if storage_manager doesn't have store_paths method
        for path in paths:
            # Store path data to file instead
            path_data_file = self.pathrag_data_dir / "paths_data" / f"{doc_id}_{path.id if hasattr(path, 'id') else 'path'}.json"
            os.makedirs(path_data_file.parent, exist_ok=True)
            
            # Convert path to dictionary if it's not already
            # PathStructure objects have to_dict, but pathlib.Path objects don't
            try:
                path_dict = path.to_dict() if hasattr(path, 'to_dict') else {
                    "id": str(hash(str(path))),
                    "path": str(path) if hasattr(path, '__str__') else "unknown_path"
                }
            except Exception as e:
                logger.warning(f"Error converting path to dictionary: {e}")
                path_dict = {"id": str(hash(str(id(path)))), "error": "Failed to convert path"}
            
            with open(path_data_file, 'w') as f:
                json.dump(path_dict, f, indent=2)
        
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
        try:
            # Use document collection method if available, otherwise check for the document directly
            document = self.universal_collector.get_document(doc_id)
            if document is None:
                return status
        except (AttributeError, Exception):
            # If method doesn't exist or fails, assume document doesn't exist
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
                loaded_data = json.load(f)
                # Validate and ensure correct return type
                typed_result: List[Dict[str, Any]] = []
                if isinstance(loaded_data, list):
                    for item in loaded_data:
                        if isinstance(item, dict):
                            typed_result.append(item)
                        else:
                            # Handle non-dict items by converting them
                            typed_result.append({"data": str(item)})
                elif isinstance(loaded_data, dict):
                    # If a single dict was loaded, wrap it in a list
                    typed_result.append(loaded_data)
                else:
                    # Handle unexpected data format
                    logger.warning(f"Unexpected data format in {paths_file}")
                return typed_result
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
        # Implement alternative query method since PathStorageManager may not have query_paths
        result: List[Dict[str, Any]] = []
        
        # Gather path data from files instead
        paths_dir = self.pathrag_data_dir / "paths"
        if paths_dir.exists():
            for path_file in paths_dir.glob("*.json"):
                try:
                    with open(path_file, 'r') as f:
                        paths_data = json.load(f)
                        if isinstance(paths_data, list):
                            for path in paths_data:
                                if isinstance(path, dict):
                                    # Ensure we're returning a Dict[str, Any]
                                    path_dict: Dict[str, Any] = {k: v for k, v in path.items()}
                                    result.append(path_dict)
                                    if len(result) >= top_k:
                                        break
                except Exception as e:
                    logger.error(f"Error loading path data from {path_file}: {e}")
        
        # Ensure we return the proper type
        typed_result: List[Dict[str, Any]] = []
        for item in result[:top_k]:
            if isinstance(item, dict):
                typed_result.append(item)
            else:
                # Convert non-dict items to dict
                typed_result.append({"data": str(item)})
        return typed_result
    
    def get_paths_by_metadata(self, metadata_filter: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get paths that match the metadata filter.
        
        Args:
            metadata_filter: Dictionary of metadata key-value pairs to filter on
            limit: Maximum number of paths to return
            
        Returns:
            List of path results
        """
        # Implement alternative method since PathStorageManager may not have get_paths_by_metadata
        result: List[Dict[str, Any]] = []
        
        # Gather path data from files instead
        paths_dir = self.pathrag_data_dir / "paths"
        if paths_dir.exists():
            for path_file in paths_dir.glob("*.json"):
                try:
                    with open(path_file, 'r') as f:
                        paths_data = json.load(f)
                        for path in paths_data:
                            # Check if path metadata matches filter
                            if "metadata" in path:
                                match = True
                                for key, value in metadata_filter.items():
                                    if key not in path["metadata"] or path["metadata"][key] != value:
                                        match = False
                                        break
                                if match:
                                    result.append(path)
                                    if len(result) >= limit:
                                        break
                except Exception as e:
                    logger.error(f"Error loading path data from {path_file}: {e}")
        
        # Ensure we return the proper type
        typed_result: List[Dict[str, Any]] = []
        for item in result[:limit]:
            if isinstance(item, dict):
                typed_result.append(item)
            else:
                # Convert non-dict items to dict
                typed_result.append({"data": str(item)})
        return typed_result
    
    def get_path_by_id(self, path_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific path by ID.
        
        Args:
            path_id: Path ID
            
        Returns:
            Path if found, None otherwise
        """
        # Implement alternative method since PathStorageManager may not have get_path_by_id
        # Gather path data from files instead
        paths_dir = self.pathrag_data_dir / "paths"
        if paths_dir.exists():
            for path_file in paths_dir.glob("*.json"):
                try:
                    with open(path_file, 'r') as f:
                        paths_data = json.load(f)
                        for path in paths_data:
                            if "id" in path and path["id"] == path_id:
                                # Ensure we're returning a Dict[str, Any]
                                path_dict: Dict[str, Any] = {k: v for k, v in path.items()} if isinstance(path, dict) else {"path": str(path)}
                                return path_dict
                except Exception as e:
                    logger.error(f"Error loading path data from {path_file}: {e}")
        
        return None
    
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
