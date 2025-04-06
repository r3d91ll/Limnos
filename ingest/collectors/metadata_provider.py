"""
Metadata Provider for Limnos Document Collectors.

This module implements the metadata provider class that provides standardized
access to document metadata, with support for extension points for different
RAG frameworks.
"""

from typing import Dict, Any, List, Optional, Set, Union
from pathlib import Path
import json
import os
from datetime import datetime

from limnos.ingest.models.metadata import UniversalMetadata
from limnos.ingest.collectors.metadata_interface import MetadataFormat, MetadataExtensionPoint


class MetadataProvider:
    """Provider for document metadata with extension points.
    
    This class provides access to universal metadata stored by the document collector,
    along with extension points for framework-specific preprocessing.
    """
    
    def __init__(self, metadata_dir: Path):
        """Initialize the metadata provider.
        
        Args:
            metadata_dir: Directory containing metadata files
        """
        self.metadata_dir = metadata_dir
        self.extension_points: Dict[str, MetadataExtensionPoint] = {}
        
        # Ensure directory exists
        os.makedirs(metadata_dir, exist_ok=True)
    
    def get_metadata(self, document_id: str, format: MetadataFormat = MetadataFormat.DICT) -> Any:
        """Get metadata for a document.
        
        Args:
            document_id: Document ID
            format: Format for the metadata
            
        Returns:
            Metadata in the requested format
        """
        metadata_path = self.metadata_dir / document_id / f"{document_id}.json"
        
        if not metadata_path.exists():
            return None
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        if format == MetadataFormat.DICT:
            return metadata
        elif format == MetadataFormat.JSON:
            return json.dumps(metadata, indent=2)
        elif format == MetadataFormat.PYDANTIC:
            return UniversalMetadata.from_dict(metadata)
        else:
            raise ValueError(f"Unsupported metadata format: {format}")
    
    def save_metadata(self, document_id: str, metadata: Union[Dict[str, Any], UniversalMetadata]) -> Path:
        """Save metadata for a document.
        
        Args:
            document_id: Document ID
            metadata: Metadata to save (dict or UniversalMetadata)
            
        Returns:
            Path to the saved metadata file
        """
        # Convert UniversalMetadata to dict if needed
        if isinstance(metadata, UniversalMetadata):
            metadata_dict = metadata.to_dict()
        else:
            metadata_dict = metadata
        
        # Ensure document directory exists
        doc_dir = self.metadata_dir / document_id
        os.makedirs(doc_dir, exist_ok=True)
        
        # Save metadata
        metadata_path = doc_dir / f"{document_id}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2)
            
        return metadata_path
    
    def get_document_ids(self) -> List[str]:
        """Get all document IDs with metadata.
        
        Returns:
            List of document IDs
        """
        if not self.metadata_dir.exists():
            return []
            
        return [d.name for d in self.metadata_dir.iterdir() 
                if d.is_dir() and (d / f"{d.name}.json").exists()]
    
    def register_extension_point(self, name: str, extension: MetadataExtensionPoint) -> None:
        """Register a metadata extension point.
        
        Args:
            name: Name of the extension point
            extension: Extension point implementation
        """
        self.extension_points[name] = extension
    
    def get_extension_point(self, name: str) -> Optional[MetadataExtensionPoint]:
        """Get a metadata extension point by name.
        
        Args:
            name: Name of the extension point
            
        Returns:
            Extension point if found, None otherwise
        """
        return self.extension_points.get(name)
    
    def list_extension_points(self) -> List[str]:
        """List all registered extension points.
        
        Returns:
            List of extension point names
        """
        return list(self.extension_points.keys())
    
    def process_metadata_with_extension(self, document_id: str, extension_name: str) -> Dict[str, Any]:
        """Process metadata with a specific extension point.
        
        Args:
            document_id: Document ID
            extension_name: Name of the extension point
            
        Returns:
            Processed metadata
            
        Raises:
            ValueError: If extension point not found or metadata not found
        """
        extension = self.get_extension_point(extension_name)
        if not extension:
            raise ValueError(f"Extension point not found: {extension_name}")
            
        metadata = self.get_metadata(document_id)
        if not metadata:
            raise ValueError(f"Metadata not found for document: {document_id}")
            
        return extension.transform_metadata(metadata)
    
    def bulk_process_metadata(self, document_ids: List[str], extension_name: str) -> Dict[str, Dict[str, Any]]:
        """Process metadata for multiple documents with a specific extension.
        
        Args:
            document_ids: List of document IDs
            extension_name: Name of the extension point
            
        Returns:
            Dictionary mapping document IDs to processed metadata
            
        Raises:
            ValueError: If extension point not found
        """
        extension = self.get_extension_point(extension_name)
        if not extension:
            raise ValueError(f"Extension point not found: {extension_name}")
            
        results = {}
        for doc_id in document_ids:
            metadata = self.get_metadata(doc_id)
            if metadata:
                results[doc_id] = extension.transform_metadata(metadata)
                
        return results
    
    def delete_metadata(self, document_id: str) -> bool:
        """Delete metadata for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if metadata was deleted, False if not found
        """
        metadata_path = self.metadata_dir / document_id / f"{document_id}.json"
        
        if not metadata_path.exists():
            return False
            
        os.remove(metadata_path)
        return True
