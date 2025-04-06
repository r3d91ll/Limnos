"""
PathRAG Metadata Preprocessor (Refactored).

This module implements the metadata preprocessor for PathRAG, transforming
universal document metadata into the format required for path extraction and processing.
This refactored version uses the new Pydantic models and metadata interfaces.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import logging
import json
from datetime import datetime

from limnos.ingest.collectors.metadata_interface import MetadataExtensionPoint
from limnos.ingest.models.metadata import DocumentType


class PathRAGMetadataPreprocessor(MetadataExtensionPoint):
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
        self.framework_name = "pathrag"
        self.output_dir = output_dir or Path("/home/todd/ML-Lab/Olympus/limnos/data/implementations/pathrag")
        self.processed_documents = {}
        self.processing_status = {}
        
        # Create output directory structure according to architectural decisions
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / "chunks", exist_ok=True)
        os.makedirs(self.output_dir / "embeddings", exist_ok=True)
        os.makedirs(self.output_dir / "paths", exist_ok=True)
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
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
        doc_id = metadata['doc_id']
        
        # Add chunking configuration
        doc_type_str = metadata.get('doc_type', DocumentType.OTHER.value)
        if isinstance(doc_type_str, str):
            try:
                doc_type = DocumentType(doc_type_str)
            except ValueError:
                doc_type = DocumentType.OTHER
        else:
            doc_type = doc_type_str
        
        pathrag_metadata['chunking_config'] = self._get_chunking_config(doc_type, metadata)
        pathrag_metadata['embedding_config'] = self._get_embedding_config(doc_type, metadata)
        pathrag_metadata['path_extraction_config'] = self._get_path_extraction_config(doc_type, metadata)
        
        # Add paths for PathRAG-specific data
        pathrag_metadata['chunks_path'] = str(self.output_dir / "chunks" / f"{doc_id}.json")
        pathrag_metadata['embeddings_path'] = str(self.output_dir / "embeddings" / f"{doc_id}.json")
        pathrag_metadata['paths_path'] = str(self.output_dir / "paths" / f"{doc_id}.json")
        
        # Add framework identification
        pathrag_metadata['framework'] = self.framework_name
        
        # Store the processed metadata
        self.processed_documents[doc_id] = pathrag_metadata
        
        # Initialize or update processing status
        if doc_id not in self.processing_status:
            self.processing_status[doc_id] = {
                'metadata_processed': True,
                'chunks_created': False,
                'embeddings_created': False,
                'paths_created': False,
                'last_updated': datetime.now().isoformat()
            }
        else:
            self.processing_status[doc_id]['metadata_processed'] = True
            self.processing_status[doc_id]['last_updated'] = datetime.now().isoformat()
        
        return pathrag_metadata
    
    def _get_chunking_config(self, doc_type: DocumentType, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get chunking configuration based on document type.
        
        Args:
            doc_type: Document type
            metadata: Universal metadata
            
        Returns:
            Chunking configuration
        """
        # Default chunking parameters
        config = {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'chunk_strategy': 'sliding_window',
            'respect_sections': True,
            'respect_paragraphs': True,
            'min_chunk_size': 100,
            'model': 'default'
        }
        
        # Customize parameters based on document type
        if doc_type == DocumentType.ACADEMIC_PAPER:
            config['chunk_size'] = 800
            config['chunk_overlap'] = 150
            config['respect_sections'] = True
            
        elif doc_type == DocumentType.DOCUMENTATION:
            config['chunk_size'] = 600
            config['chunk_overlap'] = 100
            config['respect_sections'] = True
            
        elif doc_type == DocumentType.CODE:
            config['chunk_size'] = 400
            config['chunk_overlap'] = 50
            config['respect_sections'] = False
            config['chunk_strategy'] = 'by_function'
            
        # Override with custom parameters if specified in metadata
        if 'custom' in metadata and 'chunking_params' in metadata['custom']:
            for key, value in metadata['custom']['chunking_params'].items():
                config[key] = value
        
        return config
    
    def _get_embedding_config(self, doc_type: DocumentType, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get embedding configuration based on document type.
        
        Args:
            doc_type: Document type
            metadata: Universal metadata
            
        Returns:
            Embedding configuration
        """
        # Default embedding parameters
        config = {
            'model': 'default',
            'dimensions': 768,
            'batch_size': 32,
            'include_metadata': True,
            'normalize_embeddings': True
        }
        
        # Customize parameters based on document type
        if doc_type == DocumentType.ACADEMIC_PAPER:
            config['model'] = 'scientific'
            
        elif doc_type == DocumentType.CODE:
            config['model'] = 'code'
            
        # Override with custom parameters if specified in metadata
        if 'custom' in metadata and 'embedding_params' in metadata['custom']:
            for key, value in metadata['custom']['embedding_params'].items():
                config[key] = value
        
        return config
    
    def _get_path_extraction_config(self, doc_type: DocumentType, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get path extraction configuration based on document type.
        
        Args:
            doc_type: Document type
            metadata: Universal metadata
            
        Returns:
            Path extraction configuration
        """
        # Default path extraction parameters
        config = {
            'max_path_length': 5,
            'min_path_length': 2,
            'similarity_threshold': 0.75,
            'path_types': ['sequential', 'semantic'],
            'include_document_metadata': True,
            'model': 'default'
        }
        
        # Customize parameters based on document type
        if doc_type == DocumentType.ACADEMIC_PAPER:
            config['path_types'] = ['sequential', 'semantic', 'citation']
            config['similarity_threshold'] = 0.8
            
        elif doc_type == DocumentType.DOCUMENTATION:
            config['path_types'] = ['sequential', 'semantic', 'reference']
            config['similarity_threshold'] = 0.7
            
        elif doc_type == DocumentType.CODE:
            config['path_types'] = ['sequential', 'call_graph', 'dependency']
            config['similarity_threshold'] = 0.6
            
        # Override with custom parameters if specified in metadata
        if 'custom' in metadata and 'path_params' in metadata['custom']:
            for key, value in metadata['custom']['path_params'].items():
                config[key] = value
        
        return config
    
    def get_processing_status(self, doc_id: str) -> Dict[str, Any]:
        """
        Get the processing status for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Processing status dictionary
        """
        return self.processing_status.get(doc_id, {
            'metadata_processed': False,
            'chunks_created': False,
            'embeddings_created': False,
            'paths_created': False,
            'last_updated': datetime.now().isoformat()
        })
    
    def update_processing_status(self, doc_id: str, status_updates: Dict[str, bool]) -> None:
        """
        Update the processing status for a document.
        
        Args:
            doc_id: Document ID
            status_updates: Dictionary of status updates
        """
        if doc_id not in self.processing_status:
            self.processing_status[doc_id] = {
                'metadata_processed': False,
                'chunks_created': False,
                'embeddings_created': False,
                'paths_created': False,
                'last_updated': datetime.now().isoformat()
            }
        
        # Update status fields
        for key, value in status_updates.items():
            if key in self.processing_status[doc_id]:
                self.processing_status[doc_id][key] = value
        
        # Update timestamp
        self.processing_status[doc_id]['last_updated'] = datetime.now().isoformat()
        
        # Save status to disk if document has been processed
        if doc_id in self.processed_documents:
            # Save status file alongside the paths file
            status_path = self.output_dir / "paths" / f"{doc_id}_status.json"
            with open(status_path, 'w', encoding='utf-8') as f:
                json.dump(self.processing_status[doc_id], f, indent=2)
    
    def is_fully_processed(self, doc_id: str) -> bool:
        """
        Check if a document is fully processed.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if fully processed, False otherwise
        """
        status = self.get_processing_status(doc_id)
        return (status.get('metadata_processed', False) and
                status.get('chunks_created', False) and
                status.get('embeddings_created', False) and
                status.get('paths_created', False))
    
    def get_required_metadata_fields(self) -> Set[str]:
        """
        Get the set of universal metadata fields required by this preprocessor.
        
        Returns:
            Set of field names
        """
        return {'doc_id', 'title', 'content', 'doc_type'}
    
    def get_chunking_config(self, doc_id: str) -> Dict[str, Any]:
        """
        Get chunking configuration for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Chunking configuration
        """
        # Get PathRAG-specific metadata
        if doc_id not in self.processed_documents:
            raise ValueError(f"No PathRAG metadata found for document {doc_id}")
            
        metadata = self.processed_documents[doc_id]
        
        # Return the chunking configuration
        return {
            'doc_id': doc_id,
            **metadata['chunking_config'],
            'output_path': metadata['chunks_path']
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
        if doc_id not in self.processed_documents:
            raise ValueError(f"No PathRAG metadata found for document {doc_id}")
            
        metadata = self.processed_documents[doc_id]
        
        # Return the embedding configuration
        return {
            'doc_id': doc_id,
            **metadata['embedding_config'],
            'chunks_path': metadata['chunks_path'],
            'output_path': metadata['embeddings_path']
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
        if doc_id not in self.processed_documents:
            raise ValueError(f"No PathRAG metadata found for document {doc_id}")
            
        metadata = self.processed_documents[doc_id]
        
        # Return the path extraction configuration
        return {
            'doc_id': doc_id,
            **metadata['path_extraction_config'],
            'chunks_path': metadata['chunks_path'],
            'embeddings_path': metadata['embeddings_path'],
            'output_path': metadata['paths_path']
        }
