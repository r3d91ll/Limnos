"""
Metadata Preprocessor for Limnos RAG frameworks.

This module defines the base class for metadata preprocessors that transform
universal metadata into framework-specific formats.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import json
import os
import logging

from limnos.ingest.collectors.metadata_interface import MetadataExtensionPoint


class BaseMetadataPreprocessor(MetadataExtensionPoint, ABC):
    """
    Base class for metadata preprocessors in Limnos.
    
    This class defines the common interface and functionality for all
    metadata preprocessors, which transform universal metadata into
    framework-specific formats.
    """
    
    def __init__(self, framework_name: str, output_dir: Optional[Path] = None):
        """
        Initialize the metadata preprocessor.
        
        Args:
            framework_name: Name of the framework (e.g., 'pathrag', 'graphrag')
            output_dir: Directory to store framework-specific metadata
        """
        self.framework_name = framework_name
        self.output_dir = output_dir or Path(f"/home/todd/ML-Lab/Olympus/limnos/data/implementations/{framework_name}")
        self.logger = logging.getLogger(f"{__name__}.{framework_name}")
        
        # Create output directory if it doesn't exist
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.output_dir / "metadata", exist_ok=True)
    
    def process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and transform universal metadata into framework-specific format.
        
        Args:
            metadata: Universal metadata to process
            
        Returns:
            Framework-specific metadata
        """
        # Get document ID
        doc_id = metadata.get('doc_id')
        if not doc_id:
            raise ValueError("Metadata missing doc_id")
        
        # Transform metadata
        framework_metadata = self.transform_metadata(metadata)
        
        # Add framework identifier
        framework_metadata['framework'] = self.framework_name
        framework_metadata['universal_doc_id'] = doc_id
        
        # Store framework-specific metadata if output directory is set
        if self.output_dir:
            self._store_framework_metadata(doc_id, framework_metadata)
        
        return framework_metadata
    
    @abstractmethod
    def transform_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform universal metadata into framework-specific format.
        
        This method must be implemented by each framework-specific preprocessor.
        
        Args:
            metadata: Universal metadata to transform
            
        Returns:
            Framework-specific metadata
        """
        pass
    
    def _store_framework_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> Path:
        """
        Store framework-specific metadata.
        
        Args:
            doc_id: Document ID
            metadata: Framework-specific metadata
            
        Returns:
            Path to the stored metadata file
        """
        # Create framework-specific metadata directory
        metadata_dir = self.output_dir / "metadata"
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Create metadata file path
        metadata_path = metadata_dir / f"{doc_id}.json"
        
        # Store metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Stored framework-specific metadata for {doc_id} at {metadata_path}")
        
        return metadata_path
    
    def get_framework_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get framework-specific metadata for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Framework-specific metadata or None if not found
        """
        metadata_path = self.output_dir / "metadata" / f"{doc_id}.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading framework-specific metadata for {doc_id}: {e}")
            return None
    
    def list_processed_documents(self) -> List[str]:
        """
        List all documents that have been processed by this preprocessor.
        
        Returns:
            List of document IDs
        """
        metadata_dir = self.output_dir / "metadata"
        
        if not metadata_dir.exists():
            return []
        
        return [f.stem for f in metadata_dir.glob("*.json")]
    
    def get_required_metadata_fields(self) -> Set[str]:
        """
        Get the set of universal metadata fields required by this preprocessor.
        
        Returns:
            Set of field names
        """
        return {'doc_id', 'doc_type', 'title'}
    
    def validate_universal_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """
        Validate that universal metadata contains all required fields.
        
        Args:
            metadata: Universal metadata to validate
            
        Returns:
            List of missing field names
        """
        required_fields = self.get_required_metadata_fields()
        missing_fields = [field for field in required_fields if field not in metadata]
        return missing_fields
    
    def bulk_process(self, metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple metadata records in bulk.
        
        Args:
            metadata_list: List of universal metadata records
            
        Returns:
            List of framework-specific metadata records
        """
        results = []
        for metadata in metadata_list:
            try:
                framework_metadata = self.process_metadata(metadata)
                results.append(framework_metadata)
            except Exception as e:
                doc_id = metadata.get('doc_id', 'unknown')
                self.logger.error(f"Error processing metadata for {doc_id}: {e}")
                # Add error information to results
                results.append({
                    'doc_id': doc_id,
                    'error': str(e),
                    'framework': self.framework_name
                })
        
        return results
