"""
GraphRAG Metadata Preprocessor.

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

from limnos.ingest.collectors.metadata_preprocessor import BaseMetadataPreprocessor
from limnos.ingest.collectors.metadata_schema import DocumentType


class GraphRAGMetadataPreprocessor(BaseMetadataPreprocessor):
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
        super().__init__(
            framework_name="graphrag",
            output_dir=output_dir or Path("/home/todd/ML-Lab/Olympus/limnos/data/implementations/graphrag")
        )
        
        # Create additional directories for GraphRAG-specific data
        os.makedirs(self.output_dir / "entities", exist_ok=True)
        os.makedirs(self.output_dir / "relationships", exist_ok=True)
        os.makedirs(self.output_dir / "graphs", exist_ok=True)
    
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
        
        # Add GraphRAG-specific fields
        graphrag_metadata['entity_extraction_status'] = 'pending'
        graphrag_metadata['relationship_extraction_status'] = 'pending'
        graphrag_metadata['graph_construction_status'] = 'pending'
        
        # Add paths for GraphRAG-specific data
        doc_id = metadata['doc_id']
        graphrag_metadata['entities_path'] = str(self.output_dir / "entities" / f"{doc_id}.json")
        graphrag_metadata['relationships_path'] = str(self.output_dir / "relationships" / f"{doc_id}.json")
        graphrag_metadata['graph_path'] = str(self.output_dir / "graphs" / f"{doc_id}.json")
        
        # Add GraphRAG-specific processing hints based on document type
        doc_type = metadata.get('doc_type', 'other')
        if isinstance(doc_type, str):
            try:
                doc_type = DocumentType(doc_type)
            except ValueError:
                doc_type = DocumentType.OTHER
        
        # Add extraction hints based on document type
        graphrag_metadata['extraction_hints'] = self._get_extraction_hints(doc_type, metadata)
        
        # Add graph construction parameters
        graphrag_metadata['graph_params'] = {
            'include_document_node': True,
            'merge_similar_entities': True,
            'similarity_threshold': 0.85,
            'include_section_nodes': True if metadata.get('sections') else False,
            'bidirectional_relationships': True
        }
        
        return graphrag_metadata
    
    def _get_extraction_hints(self, doc_type: DocumentType, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get extraction hints based on document type.
        
        Args:
            doc_type: Document type
            metadata: Universal metadata
            
        Returns:
            Extraction hints
        """
        hints = {
            'entity_types': ['person', 'organization', 'location', 'concept', 'term'],
            'relationship_types': ['mentions', 'related_to', 'contains', 'references'],
            'prioritize_sections': []
        }
        
        # Customize hints based on document type
        if doc_type == DocumentType.ACADEMIC_PAPER:
            hints['entity_types'].extend(['method', 'dataset', 'metric', 'result'])
            hints['relationship_types'].extend(['evaluates', 'outperforms', 'uses', 'introduces'])
            hints['prioritize_sections'] = ['abstract', 'introduction', 'conclusion', 'method']
            
        elif doc_type == DocumentType.DOCUMENTATION:
            hints['entity_types'].extend(['function', 'class', 'module', 'parameter', 'return_value'])
            hints['relationship_types'].extend(['calls', 'inherits_from', 'implements', 'returns'])
            hints['prioritize_sections'] = ['overview', 'api', 'examples']
            
        elif doc_type == DocumentType.CODE:
            hints['entity_types'].extend(['function', 'class', 'variable', 'module', 'package'])
            hints['relationship_types'].extend(['imports', 'calls', 'defines', 'inherits_from'])
            hints['prioritize_sections'] = []
            
        # Add custom entity types if specified in metadata
        if 'custom' in metadata and 'entity_types' in metadata['custom']:
            hints['entity_types'].extend(metadata['custom']['entity_types'])
            
        # Add custom relationship types if specified in metadata
        if 'custom' in metadata and 'relationship_types' in metadata['custom']:
            hints['relationship_types'].extend(metadata['custom']['relationship_types'])
        
        return hints
    
    def get_required_metadata_fields(self) -> Set[str]:
        """
        Get the set of universal metadata fields required by this preprocessor.
        
        Returns:
            Set of field names
        """
        # Add GraphRAG-specific required fields to the base required fields
        required_fields = super().get_required_metadata_fields()
        required_fields.update({'content_length', 'language'})
        return required_fields
    
    def get_entity_extraction_config(self, doc_id: str) -> Dict[str, Any]:
        """
        Get entity extraction configuration for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Entity extraction configuration
        """
        # Get GraphRAG-specific metadata
        metadata = self.get_framework_metadata(doc_id)
        if not metadata:
            raise ValueError(f"No GraphRAG metadata found for document {doc_id}")
        
        # Extract entity extraction configuration
        return {
            'doc_id': doc_id,
            'entity_types': metadata['extraction_hints']['entity_types'],
            'prioritize_sections': metadata['extraction_hints']['prioritize_sections'],
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
        metadata = self.get_framework_metadata(doc_id)
        if not metadata:
            raise ValueError(f"No GraphRAG metadata found for document {doc_id}")
        
        # Extract relationship extraction configuration
        return {
            'doc_id': doc_id,
            'relationship_types': metadata['extraction_hints']['relationship_types'],
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
        metadata = self.get_framework_metadata(doc_id)
        if not metadata:
            raise ValueError(f"No GraphRAG metadata found for document {doc_id}")
        
        # Extract graph construction configuration
        return {
            'doc_id': doc_id,
            'entities_path': metadata['entities_path'],
            'relationships_path': metadata['relationships_path'],
            'output_path': metadata['graph_path'],
            'include_document_node': metadata['graph_params']['include_document_node'],
            'merge_similar_entities': metadata['graph_params']['merge_similar_entities'],
            'similarity_threshold': metadata['graph_params']['similarity_threshold'],
            'include_section_nodes': metadata['graph_params']['include_section_nodes'],
            'bidirectional_relationships': metadata['graph_params']['bidirectional_relationships']
        }
    
    def update_processing_status(self, doc_id: str, status_updates: Dict[str, bool]) -> None:
        """
        Update the processing status for a document.
        
        Args:
            doc_id: Document ID
            status_updates: Dictionary of status updates
        """
        # The implementation maintains backward compatibility with the old signature
        # while conforming to the new interface
        # Get GraphRAG-specific metadata
        metadata_path = self.output_dir / "metadata" / f"{doc_id}.json"
        if not metadata_path.exists():
            raise ValueError(f"No GraphRAG metadata found for document {doc_id}")
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Update status for each field in the status_updates dictionary
        for key, value in status_updates.items():
            # Convert key to the expected status field format
            status_field = f"{key}_status" if not key.endswith('_status') else key
            if status_field in metadata:
                # Convert boolean value to status string for backward compatibility
                status_str = 'completed' if value else 'pending'
                metadata[status_field] = status_str
            
            # Add timestamp for each updated field
            timestamp_field = f"{key.replace('_status', '')}_timestamp"
        metadata[timestamp_field] = datetime.now().isoformat()
        
        # Save updated metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Log status updates in the new format
        status_items = ", ".join([f"{k}: {v}" for k, v in status_updates.items()])
        logging.info(f"Updated status for document {doc_id}: {status_items}")
        
    def get_processing_status(self, doc_id: str) -> Dict[str, str]:
        """
        Get the processing status for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dictionary with processing status for each stage
        """
        # Get GraphRAG-specific metadata
        metadata = self.get_framework_metadata(doc_id)
        if not metadata:
            return {
                'entity_extraction': 'unknown',
                'relationship_extraction': 'unknown',
                'graph_construction': 'unknown'
            }
            
        # Extract status fields
        return {
            'entity_extraction': metadata.get('entity_extraction_status', 'unknown'),
            'relationship_extraction': metadata.get('relationship_extraction_status', 'unknown'),
            'graph_construction': metadata.get('graph_construction_status', 'unknown')
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
