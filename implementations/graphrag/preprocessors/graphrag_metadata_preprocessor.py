"""
GraphRAG Metadata Preprocessor (Refactored).

This module implements the metadata preprocessor for GraphRAG, transforming
universal document metadata into the format required for entity extraction,
relationship identification, and graph construction.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import logging
import json
from datetime import datetime

from limnos.ingest.collectors.metadata_interface import MetadataExtensionPoint
from limnos.ingest.models.metadata import DocumentType


class GraphRAGMetadataPreprocessor(MetadataExtensionPoint):
    """
    Metadata preprocessor for GraphRAG.
    
    This preprocessor transforms universal document metadata into the format
    required for GraphRAG's entity extraction, relationship identification,
    and graph construction components.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the GraphRAG metadata preprocessor.
        
        Args:
            output_dir: Directory to store GraphRAG-specific metadata
        """
        self.framework_name = "graphrag"
        self.output_dir = output_dir or Path("/home/todd/ML-Lab/Olympus/limnos/data/implementations/graphrag")
        self.processed_documents: Dict[str, Dict[str, Any]] = {}
        self.processing_status: Dict[str, Dict[str, bool]] = {}
        
        # Create output directory structure according to architectural decisions
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / "entities", exist_ok=True)
        os.makedirs(self.output_dir / "relationships", exist_ok=True)
        os.makedirs(self.output_dir / "graphs", exist_ok=True)
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def transform_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform universal metadata into GraphRAG-specific format.
        
        Args:
            metadata: Universal metadata to transform
            
        Returns:
            GraphRAG-specific metadata
        """
        # Start with a copy of the universal metadata
        graphrag_metadata = metadata.copy()
        
        # Add GraphRAG-specific configuration
        doc_id = metadata['doc_id']
        
        # Add entity extraction configuration
        graphrag_metadata['entity_extraction_config'] = {
            'model': 'default',
            'confidence_threshold': 0.7,
            'max_entities_per_document': 50,
            'entity_types': ['person', 'organization', 'location', 'concept', 'term'],
            'prioritize_sections': self._get_priority_sections(metadata)
        }
        
        # Add relationship extraction configuration
        graphrag_metadata['relationship_extraction_config'] = {
            'model': 'default',
            'confidence_threshold': 0.6,
            'max_relationships_per_document': 100,
            'relationship_types': ['mentions', 'related_to', 'contains', 'references'],
            'bidirectional_relationships': True
        }
        
        # Add graph construction configuration
        graphrag_metadata['graph_construction_config'] = {
            'merge_similar_entities': True,
            'similarity_threshold': 0.85,
            'include_document_node': True,
            'include_section_nodes': True if metadata.get('sections') else False
        }
        
        # Add paths for GraphRAG-specific data
        graphrag_metadata['entities_path'] = str(self.output_dir / "entities" / f"{doc_id}.json")
        graphrag_metadata['relationships_path'] = str(self.output_dir / "relationships" / f"{doc_id}.json")
        graphrag_metadata['graph_path'] = str(self.output_dir / "graphs" / f"{doc_id}.json")
        
        # Add framework identification
        graphrag_metadata['framework'] = self.framework_name
        
        # Store the processed metadata
        self.processed_documents[doc_id] = graphrag_metadata
        
        # Initialize or update processing status
        if doc_id not in self.processing_status:
            # Create a status dictionary with proper typing
            # Store the timestamp in a separate field to avoid type conflicts
            self.processing_status[doc_id] = {
                'metadata_processed': True,
                'entities_extracted': False,
                'relationships_extracted': False,
                'graph_constructed': False
            }
            
            # Store timestamp separately in attributes
            self.processed_documents.setdefault(doc_id, {})['last_updated'] = datetime.now().isoformat()
        else:
            self.processing_status[doc_id]['metadata_processed'] = True
            
            # Store timestamp separately in attributes
            self.processed_documents.setdefault(doc_id, {})['last_updated'] = datetime.now().isoformat()
        
        return graphrag_metadata
    
    def _get_priority_sections(self, metadata: Dict[str, Any]) -> List[str]:
        """
        Get priority sections based on document type.
        
        Args:
            metadata: Universal metadata
            
        Returns:
            List of priority section names
        """
        # Get document type
        doc_type_str = metadata.get('doc_type', DocumentType.OTHER.value)
        
        # Convert string to DocumentType if needed
        if isinstance(doc_type_str, str):
            try:
                doc_type = DocumentType(doc_type_str)
            except ValueError:
                doc_type = DocumentType.OTHER
        else:
            doc_type = doc_type_str
            
        # Define priority sections based on document type
        if doc_type == DocumentType.ACADEMIC_PAPER:
            return ['abstract', 'introduction', 'conclusion', 'method']
        elif doc_type == DocumentType.DOCUMENTATION:
            return ['overview', 'api', 'examples']
        elif doc_type == DocumentType.CODE:
            return []
        else:
            return []
    
    def get_processing_status(self, doc_id: str) -> Dict[str, Any]:
        """
        Get the processing status for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Processing status dictionary
        """
        # Create a default status dictionary with proper typing
        # Create a default status dictionary with boolean values only
        default_status: Dict[str, bool] = {
            'metadata_processed': False,
            'entities_extracted': False,
            'relationships_extracted': False,
            'graph_constructed': False
        }
        
        # Get the status with explicit typing to ensure proper return type
        result: Dict[str, Any] = self.processing_status.get(doc_id, default_status)
        return result
    
    def update_processing_status(self, doc_id: str, status_updates: Dict[str, bool]) -> None:
        """
        Update the processing status for a document.
        
        Args:
            doc_id: Document ID
            status_updates: Dictionary of status updates
        """
        if doc_id not in self.processing_status:
            # Initialize with proper typing - only boolean values
            self.processing_status[doc_id] = {
                'metadata_processed': False,
                'entities_extracted': False,
                'relationships_extracted': False,
                'graph_constructed': False
            }
            # Store timestamp separately in attributes
            self.processed_documents.setdefault(doc_id, {})['last_updated'] = datetime.now().isoformat()
        
        # Update status fields - only update valid boolean fields
        for key, value in status_updates.items():
            if key in self.processing_status[doc_id] and key != 'last_updated':
                # Ensure we're setting boolean values
                self.processing_status[doc_id][key] = bool(value)
        
        # Update timestamp separately
        # Store timestamp in document attributes instead
        self.processed_documents.setdefault(doc_id, {})['last_updated'] = datetime.now().isoformat()
        
        # Save status to disk if document has been processed
        if doc_id in self.processed_documents:
            # Save status file alongside the graph file
            status_path = self.output_dir / "graphs" / f"{doc_id}_status.json"
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
        # Explicitly evaluate each condition and cast to bool to ensure proper type
        metadata_processed = bool(status.get('metadata_processed', False))
        entities_extracted = bool(status.get('entities_extracted', False))
        relationships_extracted = bool(status.get('relationships_extracted', False))
        graph_constructed = bool(status.get('graph_constructed', False))
        
        # Return a properly typed boolean result
        result: bool = (metadata_processed and entities_extracted and 
                       relationships_extracted and graph_constructed)
        return result
    
    def get_required_metadata_fields(self) -> Set[str]:
        """
        Get the set of universal metadata fields required by this preprocessor.
        
        Returns:
            Set of field names
        """
        return {'doc_id', 'title', 'content', 'doc_type'}
    
    def get_entity_extraction_config(self, doc_id: str) -> Dict[str, Any]:
        """
        Get entity extraction configuration for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Entity extraction configuration
        """
        # Get GraphRAG-specific metadata
        if doc_id not in self.processed_documents:
            raise ValueError(f"No GraphRAG metadata found for document {doc_id}")
            
        metadata = self.processed_documents[doc_id]
        
        # Return the entity extraction configuration
        return {
            'doc_id': doc_id,
            **metadata['entity_extraction_config'],
            'output_path': metadata['entities_path']
        }
    
    def get_relationship_extraction_config(self, doc_id: str) -> Dict[str, Any]:
        """
        Get relationship extraction configuration for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Relationship extraction configuration
        """
        # Get GraphRAG-specific metadata
        if doc_id not in self.processed_documents:
            raise ValueError(f"No GraphRAG metadata found for document {doc_id}")
            
        metadata = self.processed_documents[doc_id]
        
        # Return the relationship extraction configuration
        return {
            'doc_id': doc_id,
            **metadata['relationship_extraction_config'],
            'entities_path': metadata['entities_path'],
            'output_path': metadata['relationships_path']
        }
    
    def get_graph_construction_config(self, doc_id: str) -> Dict[str, Any]:
        """
        Get graph construction configuration for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Graph construction configuration
        """
        # Get GraphRAG-specific metadata
        if doc_id not in self.processed_documents:
            raise ValueError(f"No GraphRAG metadata found for document {doc_id}")
            
        metadata = self.processed_documents[doc_id]
        
        # Return the graph construction configuration
        return {
            'doc_id': doc_id,
            **metadata['graph_construction_config'],
            'entities_path': metadata['entities_path'],
            'relationships_path': metadata['relationships_path'],
            'output_path': metadata['graph_path']
        }
