"""
Universal Document Collector Integration for PathRAG

This module provides integration components to connect PathRAG with the
Universal Document Collector, handling document processing, path extraction,
and storage while maintaining the separation between universal and
framework-specific metadata.
"""

import os
import json
import logging
import time
from pathlib import Path as FilePath
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
from ..core.path_structures import Path, PathNode, PathEdge, PathIndex
from ..core.path_vector_store import PathVectorStore
from ..core.path_storage_manager import PathStorageManager

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
        
        # Source directories (universal document storage)
        self.source_documents_dir = self.config.get(
            'source_documents_dir', 
            os.path.join('/home/todd/ML-Lab/Olympus/limnos/data/source_documents')
        )
        
        # Base directory for PathRAG-specific data
        self.pathrag_data_dir = self.config.get(
            'pathrag_data_dir',
            os.path.join('/home/todd/ML-Lab/Olympus/limnos/data/implementations/pathrag')
        )
        
        # Directory for processed document metadata
        self.processed_docs_dir = os.path.join(self.pathrag_data_dir, 'processed_documents')
        
        # Directory for processing logs
        self.logs_dir = os.path.join(self.pathrag_data_dir, 'logs')
        
        # Create necessary directories
        for directory in [self.processed_docs_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize PathRAG components
        self.extraction_pipeline = ExtractionPipeline(
            self.config.get('extraction_pipeline_config', {})
        )
        
        # Initialize storage manager
        self.storage_manager = PathStorageManager(
            self.config.get('storage_manager_config', {})
        )
        
        # Track processed documents
        self.processed_document_ids = self._load_processed_document_ids()
        
        # Batch size for processing
        self.batch_size = self.config.get('batch_size', 10)
        
        # Processing timeout per document (seconds)
        self.processing_timeout = self.config.get('processing_timeout', 300)
        
        # Skip document types that aren't supported
        self.supported_extensions = self.config.get('supported_extensions', 
                                                   ['.txt', '.md', '.pdf', '.json'])
        
        # Optional processing hooks
        self.pre_processing_hook = None
        self.post_processing_hook = None
    
    def _load_processed_document_ids(self) -> Set[str]:
        """
        Load the set of already processed document IDs.
        
        Returns:
            Set of processed document IDs
        """
        processed_ids = set()
        
        # Check which documents have already been processed
        if os.path.exists(self.processed_docs_dir):
            for filename in os.listdir(self.processed_docs_dir):
                if filename.endswith('.json'):
                    processed_ids.add(filename[:-5])  # Remove .json extension
        
        logger.info(f"Loaded {len(processed_ids)} processed document IDs")
        return processed_ids
    
    def _save_processed_document_id(self, document_id: str, metadata: Dict[str, Any]) -> None:
        """
        Save record of a processed document.
        
        Args:
            document_id: Document ID
            metadata: Processing metadata
        """
        # Create a record of processing
        filepath = os.path.join(self.processed_docs_dir, f"{document_id}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Add to processed set
        self.processed_document_ids.add(document_id)
    
    def _get_document_content(self, document_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        Get document content and universal metadata.
        
        Args:
            document_id: Document ID
            
        Returns:
            Tuple of (document_content, universal_metadata)
            
        Raises:
            DocumentProcessingError: If document or metadata not found
        """
        # Try to find document file
        document_path = None
        metadata_path = os.path.join(self.source_documents_dir, f"{document_id}.json")
        
        # Check for various document formats
        for ext in self.supported_extensions:
            path = os.path.join(self.source_documents_dir, f"{document_id}{ext}")
            if os.path.exists(path):
                document_path = path
                break
        
        # Check if document exists
        if not document_path:
            raise DocumentProcessingError(f"Document file not found for ID: {document_id}")
        
        # Check if metadata exists
        if not os.path.exists(metadata_path):
            raise DocumentProcessingError(f"Universal metadata not found for ID: {document_id}")
        
        # Load universal metadata
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            raise DocumentProcessingError(f"Error loading metadata: {str(e)}")
        
        # Load document content
        try:
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise DocumentProcessingError(f"Error loading document: {str(e)}")
        
        return content, metadata
    
    def set_pre_processing_hook(self, hook: Callable[[str, Dict[str, Any]], Dict[str, Any]]) -> None:
        """
        Set a hook to run before processing a document.
        
        Args:
            hook: Function that takes (document_id, metadata) and returns metadata
        """
        self.pre_processing_hook = hook
    
    def set_post_processing_hook(self, hook: Callable[[str, Dict[str, Any], List[Path]], Dict[str, Any]]) -> None:
        """
        Set a hook to run after processing a document.
        
        Args:
            hook: Function that takes (document_id, metadata, paths) and returns metadata
        """
        self.post_processing_hook = hook
    
    def process_document(self, document_id: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process a single document from the Universal Document Collector.
        
        Args:
            document_id: Document ID
            force_reprocess: Whether to reprocess even if already processed
            
        Returns:
            Processing results metadata
            
        Raises:
            DocumentProcessingError: If processing fails
        """
        # Check if already processed
        if document_id in self.processed_document_ids and not force_reprocess:
            logger.info(f"Document {document_id} already processed, skipping")
            return {"status": "skipped", "document_id": document_id}
        
        processing_start_time = time.time()
        processing_metadata = {
            "document_id": document_id,
            "processing_started": datetime.now().isoformat(),
            "status": "processing"
        }
        
        try:
            # Get document content and metadata
            content, universal_metadata = self._get_document_content(document_id)
            
            # Call pre-processing hook if set
            if self.pre_processing_hook:
                universal_metadata = self.pre_processing_hook(document_id, universal_metadata)
            
            # Process document through extraction pipeline
            logger.info(f"Processing document {document_id}")
            
            # Create document object with metadata
            document = {
                "id": document_id,
                "content": content,
                "metadata": universal_metadata
            }
            
            # Process through pipeline with timeout
            start_time = time.time()
            result = self.extraction_pipeline.process_document(document)
            processing_time = time.time() - start_time
            
            # Extract paths
            paths = result.get('paths', [])
            
            # Store paths
            stored_paths = []
            for path in paths:
                # Ensure document_id is in node metadata
                for node in path.nodes:
                    if not node.metadata:
                        node.metadata = {}
                    node.metadata['document_id'] = document_id
                
                # Save path
                self.storage_manager.save_path(path)
                stored_paths.append(path)
            
            # Call post-processing hook if set
            if self.post_processing_hook:
                universal_metadata = self.post_processing_hook(document_id, universal_metadata, stored_paths)
            
            # Update processing metadata
            processing_metadata.update({
                "status": "success",
                "processing_completed": datetime.now().isoformat(),
                "processing_time_seconds": processing_time,
                "path_count": len(stored_paths),
                "entity_count": len(result.get('entities', [])),
                "relationship_count": len(result.get('relationships', []))
            })
            
            # Save processing record
            self._save_processed_document_id(document_id, processing_metadata)
            
            logger.info(f"Successfully processed document {document_id} - extracted {len(stored_paths)} paths")
            return processing_metadata
            
        except Exception as e:
            # Log the exception
            logger.error(f"Error processing document {document_id}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Update processing metadata
            processing_metadata.update({
                "status": "error",
                "error_message": str(e),
                "processing_completed": datetime.now().isoformat(),
                "processing_time_seconds": time.time() - processing_start_time
            })
            
            # Still save the processing record to avoid repeated failures
            self._save_processed_document_id(document_id, processing_metadata)
            
            # Re-raise as DocumentProcessingError
            raise DocumentProcessingError(f"Failed to process document {document_id}: {str(e)}")
    
    def process_documents(
        self, 
        document_ids: List[str] = None,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Process multiple documents from the Universal Document Collector.
        
        Args:
            document_ids: List of document IDs to process (if None, process all unprocessed)
            force_reprocess: Whether to reprocess documents even if already processed
            
        Returns:
            Processing results metadata
        """
        start_time = time.time()
        
        # If no document IDs provided, process all unprocessed documents
        if document_ids is None:
            document_ids = self._get_unprocessed_document_ids()
        
        # Process in batches
        results = {
            "total": len(document_ids),
            "success": 0,
            "error": 0,
            "skipped": 0,
            "documents": {}
        }
        
        for i in range(0, len(document_ids), self.batch_size):
            batch = document_ids[i:i+self.batch_size]
            
            for doc_id in batch:
                try:
                    result = self.process_document(doc_id, force_reprocess)
                    results["documents"][doc_id] = result
                    
                    if result["status"] == "success":
                        results["success"] += 1
                    elif result["status"] == "skipped":
                        results["skipped"] += 1
                    else:
                        results["error"] += 1
                        
                except DocumentProcessingError as e:
                    logger.error(f"Error processing document {doc_id}: {str(e)}")
                    results["documents"][doc_id] = {
                        "status": "error",
                        "error_message": str(e)
                    }
                    results["error"] += 1
        
        # Save index and vector store
        try:
            self.storage_manager.save_index()
            self.storage_manager.save_vector_store()
        except Exception as e:
            logger.error(f"Error saving index or vector store: {str(e)}")
            results["index_save_error"] = str(e)
        
        # Update processing metadata
        results["processing_time_seconds"] = time.time() - start_time
        
        return results
    
    def _get_unprocessed_document_ids(self) -> List[str]:
        """
        Get IDs of all unprocessed documents in the source directory.
        
        Returns:
            List of unprocessed document IDs
        """
        all_document_ids = set()
        
        # Scan source directory for metadata files
        if os.path.exists(self.source_documents_dir):
            for filename in os.listdir(self.source_documents_dir):
                if filename.endswith('.json'):
                    document_id = filename[:-5]  # Remove .json extension
                    all_document_ids.add(document_id)
        
        # Filter out already processed documents
        unprocessed_ids = list(all_document_ids - self.processed_document_ids)
        logger.info(f"Found {len(unprocessed_ids)} unprocessed documents")
        
        return unprocessed_ids
    
    def get_processing_status(self, document_id: str) -> Dict[str, Any]:
        """
        Get processing status for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Processing status metadata or None if not processed
        """
        if document_id not in self.processed_document_ids:
            return None
        
        filepath = os.path.join(self.processed_docs_dir, f"{document_id}.json")
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading processing status for {document_id}: {str(e)}")
            return None
    
    def check_for_updates(self) -> List[str]:
        """
        Check for updated documents in the source directory.
        
        Returns:
            List of document IDs that have been updated
        """
        updated_docs = []
        
        # Check all processed documents for updates
        for doc_id in self.processed_document_ids:
            # Get processing metadata
            processing_meta = self.get_processing_status(doc_id)
            
            if not processing_meta:
                continue
            
            # Check document file and metadata
            metadata_path = os.path.join(self.source_documents_dir, f"{doc_id}.json")
            
            if not os.path.exists(metadata_path):
                # Document no longer exists
                continue
            
            # Check if metadata has been modified since processing
            try:
                metadata_mtime = os.path.getmtime(metadata_path)
                processing_time = datetime.fromisoformat(
                    processing_meta.get("processing_completed", "2000-01-01T00:00:00")
                ).timestamp()
                
                if metadata_mtime > processing_time:
                    # Metadata has been updated
                    updated_docs.append(doc_id)
            except Exception as e:
                logger.error(f"Error checking updates for {doc_id}: {str(e)}")
        
        logger.info(f"Found {len(updated_docs)} updated documents")
        return updated_docs
    
    def process_updated_documents(self) -> Dict[str, Any]:
        """
        Process all documents that have been updated since last processing.
        
        Returns:
            Processing results metadata
        """
        updated_docs = self.check_for_updates()
        return self.process_documents(updated_docs, force_reprocess=True)
    
    def reprocess_all_documents(self) -> Dict[str, Any]:
        """
        Reprocess all documents in the source directory.
        
        Returns:
            Processing results metadata
        """
        # Get all document IDs
        document_ids = []
        
        if os.path.exists(self.source_documents_dir):
            for filename in os.listdir(self.source_documents_dir):
                if filename.endswith('.json'):
                    document_id = filename[:-5]  # Remove .json extension
                    document_ids.append(document_id)
        
        logger.info(f"Reprocessing all {len(document_ids)} documents")
        return self.process_documents(document_ids, force_reprocess=True)
    
    def reprocess_failed_documents(self) -> Dict[str, Any]:
        """
        Reprocess all documents that failed processing.
        
        Returns:
            Processing results metadata
        """
        failed_docs = []
        
        # Check all processed documents for failures
        for doc_id in self.processed_document_ids:
            # Get processing metadata
            processing_meta = self.get_processing_status(doc_id)
            
            if processing_meta and processing_meta.get("status") == "error":
                failed_docs.append(doc_id)
        
        logger.info(f"Reprocessing {len(failed_docs)} failed documents")
        return self.process_documents(failed_docs, force_reprocess=True)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about document processing.
        
        Returns:
            Dictionary of processing statistics
        """
        total_processed = len(self.processed_document_ids)
        successful = 0
        failed = 0
        paths_count = 0
        
        # Check all processed documents
        for doc_id in self.processed_document_ids:
            # Get processing metadata
            processing_meta = self.get_processing_status(doc_id)
            
            if not processing_meta:
                continue
            
            if processing_meta.get("status") == "success":
                successful += 1
                paths_count += processing_meta.get("path_count", 0)
            elif processing_meta.get("status") == "error":
                failed += 1
        
        # Get storage statistics
        storage_stats = self.storage_manager.get_statistics()
        
        return {
            "total_processed": total_processed,
            "successful": successful,
            "failed": failed,
            "paths_extracted": paths_count,
            "storage_stats": storage_stats
        }
    
    def clear_processing_records(self, document_ids: List[str] = None) -> int:
        """
        Clear processing records for specified documents.
        
        Args:
            document_ids: List of document IDs to clear (if None, clear all)
            
        Returns:
            Number of records cleared
        """
        if document_ids is None:
            # Clear all records
            document_ids = list(self.processed_document_ids)
        
        cleared_count = 0
        
        for doc_id in document_ids:
            filepath = os.path.join(self.processed_docs_dir, f"{doc_id}.json")
            
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    self.processed_document_ids.remove(doc_id)
                    cleared_count += 1
                except Exception as e:
                    logger.error(f"Error clearing record for {doc_id}: {str(e)}")
        
        logger.info(f"Cleared {cleared_count} processing records")
        return cleared_count
