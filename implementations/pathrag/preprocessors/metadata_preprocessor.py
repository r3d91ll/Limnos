"""
PathRAG Metadata Preprocessor.

This module implements the metadata preprocessor for PathRAG, transforming
universal document metadata into the format required for path extraction and processing.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import logging
import json
from datetime import datetime

from limnos.ingest.collectors.metadata_preprocessor import BaseMetadataPreprocessor
from limnos.ingest.collectors.metadata_schema import DocumentType


class PathRAGMetadataPreprocessor(BaseMetadataPreprocessor):
    """
    Metadata preprocessor for PathRAG.
    
    This preprocessor transforms universal document metadata into the format
    required for PathRAG's path extraction and processing components.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the PathRAG metadata preprocessor.
        
        Args:
            output_dir: Directory to store PathRAG-specific metadata
        """
        super().__init__(
            framework_name="pathrag",
            output_dir=output_dir or Path("/home/todd/ML-Lab/Olympus/limnos/data/implementations/pathrag")
        )
        
        # Create additional directories for PathRAG-specific data
        os.makedirs(self.output_dir / "chunks", exist_ok=True)
        os.makedirs(self.output_dir / "embeddings", exist_ok=True)
        os.makedirs(self.output_dir / "paths", exist_ok=True)
    
    def transform_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform universal metadata into PathRAG-specific format.
        
        Args:
            metadata: Universal metadata to transform
            
        Returns:
            PathRAG-specific metadata
        """
        # Start with a copy of the universal metadata
        pathrag_metadata = metadata.copy()
        
        # Add PathRAG-specific fields
        pathrag_metadata['chunking_status'] = 'pending'
        pathrag_metadata['embedding_status'] = 'pending'
        pathrag_metadata['path_extraction_status'] = 'pending'
        
        # Add paths for PathRAG-specific data
        doc_id = metadata['doc_id']
        pathrag_metadata['chunks_path'] = str(self.output_dir / "chunks" / f"{doc_id}.json")
        pathrag_metadata['embeddings_path'] = str(self.output_dir / "embeddings" / f"{doc_id}.json")
        pathrag_metadata['paths_path'] = str(self.output_dir / "paths" / f"{doc_id}.json")
        
        # Add PathRAG-specific processing hints based on document type
        doc_type = metadata.get('doc_type', 'other')
        if isinstance(doc_type, str):
            try:
                doc_type = DocumentType(doc_type)
            except ValueError:
                doc_type = DocumentType.OTHER
        
        # Add chunking and path extraction parameters
        pathrag_metadata['chunking_params'] = self._get_chunking_params(doc_type, metadata)
        pathrag_metadata['path_params'] = self._get_path_params(doc_type, metadata)
        
        return pathrag_metadata
    
    def _get_chunking_params(self, doc_type: DocumentType, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get chunking parameters based on document type.
        
        Args:
            doc_type: Document type
            metadata: Universal metadata
            
        Returns:
            Chunking parameters
        """
        # Default chunking parameters
        params = {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'chunk_strategy': 'sliding_window',
            'respect_sections': True,
            'respect_paragraphs': True,
            'min_chunk_size': 100
        }
        
        # Customize parameters based on document type
        if doc_type == DocumentType.ACADEMIC_PAPER:
            params['chunk_size'] = 800
            params['chunk_overlap'] = 150
            params['respect_sections'] = True
            
        elif doc_type == DocumentType.DOCUMENTATION:
            params['chunk_size'] = 600
            params['chunk_overlap'] = 100
            params['respect_sections'] = True
            
        elif doc_type == DocumentType.CODE:
            params['chunk_size'] = 400
            params['chunk_overlap'] = 50
            params['respect_sections'] = False
            params['chunk_strategy'] = 'by_function'
            
        # Override with custom parameters if specified in metadata
        if 'custom' in metadata and 'chunking_params' in metadata['custom']:
            for key, value in metadata['custom']['chunking_params'].items():
                params[key] = value
        
        return params
    
    def _get_path_params(self, doc_type: DocumentType, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get path extraction parameters based on document type.
        
        Args:
            doc_type: Document type
            metadata: Universal metadata
            
        Returns:
            Path extraction parameters
        """
        # Default path extraction parameters
        params = {
            'max_path_length': 5,
            'min_path_length': 2,
            'similarity_threshold': 0.75,
            'path_types': ['sequential', 'semantic'],
            'include_document_metadata': True
        }
        
        # Customize parameters based on document type
        if doc_type == DocumentType.ACADEMIC_PAPER:
            params['path_types'] = ['sequential', 'semantic', 'citation']
            params['similarity_threshold'] = 0.8
            
        elif doc_type == DocumentType.DOCUMENTATION:
            params['path_types'] = ['sequential', 'semantic', 'reference']
            params['similarity_threshold'] = 0.7
            
        elif doc_type == DocumentType.CODE:
            params['path_types'] = ['sequential', 'call_graph', 'dependency']
            params['similarity_threshold'] = 0.6
            
        # Override with custom parameters if specified in metadata
        if 'custom' in metadata and 'path_params' in metadata['custom']:
            for key, value in metadata['custom']['path_params'].items():
                params[key] = value
        
        return params
    
    def get_required_metadata_fields(self) -> Set[str]:
        """
        Get the set of universal metadata fields required by this preprocessor.
        
        Returns:
            Set of field names
        """
        # Add PathRAG-specific required fields to the base required fields
        required_fields = super().get_required_metadata_fields()
        required_fields.update({'content_length', 'language'})
        return required_fields
    
    def get_chunking_config(self, doc_id: str) -> Dict[str, Any]:
        """
        Get chunking configuration for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Chunking configuration
        """
        # Get PathRAG-specific metadata
        metadata = self.get_framework_metadata(doc_id)
        if not metadata:
            raise ValueError(f"No PathRAG metadata found for document {doc_id}")
        
        # Extract chunking configuration
        return {
            'doc_id': doc_id,
            'output_path': metadata['chunks_path'],
            **metadata['chunking_params']
        }
    
    def get_embedding_config(self, doc_id: str) -> Dict[str, Any]:
        """
        Get embedding configuration for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Embedding configuration
        """
        # Get PathRAG-specific metadata
        metadata = self.get_framework_metadata(doc_id)
        if not metadata:
            raise ValueError(f"No PathRAG metadata found for document {doc_id}")
        
        # Extract embedding configuration
        return {
            'doc_id': doc_id,
            'chunks_path': metadata['chunks_path'],
            'output_path': metadata['embeddings_path'],
            'model_name': 'sentence-transformers/all-mpnet-base-v2',  # Default model
            'batch_size': 32
        }
    
    def get_path_extraction_config(self, doc_id: str) -> Dict[str, Any]:
        """
        Get path extraction configuration for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Path extraction configuration
        """
        # Get PathRAG-specific metadata
        metadata = self.get_framework_metadata(doc_id)
        if not metadata:
            raise ValueError(f"No PathRAG metadata found for document {doc_id}")
        
        # Extract path extraction configuration
        return {
            'doc_id': doc_id,
            'chunks_path': metadata['chunks_path'],
            'embeddings_path': metadata['embeddings_path'],
            'output_path': metadata['paths_path'],
            **metadata['path_params']
        }
    
    def update_processing_status(self, doc_id: str, stage: str, status: str) -> None:
        """
        Update the processing status for a document.
        
        Args:
            doc_id: Document ID
            stage: Processing stage ('chunking', 'embedding', 'path_extraction')
            status: Status ('pending', 'in_progress', 'completed', 'failed')
        """
        # Get PathRAG-specific metadata
        metadata_path = self.output_dir / "metadata" / f"{doc_id}.json"
        if not metadata_path.exists():
            raise ValueError(f"No PathRAG metadata found for document {doc_id}")
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Update status
        status_field = f"{stage}_status"
        if status_field in metadata:
            metadata[status_field] = status
            
        # Add timestamp
        timestamp_field = f"{stage}_timestamp"
        metadata[timestamp_field] = datetime.now().isoformat()
        
        # Save updated metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Updated {stage} status to {status} for document {doc_id}")
        
    def get_document_chunks(self, doc_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get chunks for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunks or None if not found
        """
        # Get PathRAG-specific metadata
        metadata = self.get_framework_metadata(doc_id)
        if not metadata:
            return None
        
        # Check if chunks exist
        chunks_path = Path(metadata['chunks_path'])
        if not chunks_path.exists():
            return None
        
        # Load chunks
        try:
            with open(chunks_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading chunks for {doc_id}: {e}")
            return None
            
    def get_processing_status(self, doc_id: str) -> Dict[str, str]:
        """
        Get the processing status for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dictionary with processing status for each stage
        """
        # Get PathRAG-specific metadata
        metadata = self.get_framework_metadata(doc_id)
        if not metadata:
            return {
                'chunking': 'unknown',
                'embedding': 'unknown',
                'path_extraction': 'unknown'
            }
            
        # Extract status fields
        return {
            'chunking': metadata.get('chunking_status', 'unknown'),
            'embedding': metadata.get('embedding_status', 'unknown'),
            'path_extraction': metadata.get('path_extraction_status', 'unknown')
        }
        
    def is_document_processed(self, doc_id: str) -> bool:
        """
        Check if a document has been fully processed.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document is fully processed, False otherwise
        """
        status = self.get_processing_status(doc_id)
        return all(s == 'completed' for s in status.values())
