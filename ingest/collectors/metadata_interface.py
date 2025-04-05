"""
Metadata Interface for Limnos Document Collectors.

This module defines the interfaces for standardized metadata handling across
different RAG frameworks, enabling clear separation between universal and
framework-specific metadata while providing extension points.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Protocol, Callable, TypeVar, Generic, Union
from pathlib import Path
import json
from enum import Enum

from limnos.ingest.collectors.metadata_schema import UniversalMetadataSchema, DocumentType


class MetadataFormat(Enum):
    """Enum defining the metadata formats supported by the interface."""
    JSON = "json"
    DICT = "dict"
    SCHEMA = "schema"


class MetadataExtensionPoint(Protocol):
    """Protocol defining the interface for metadata extension points."""
    
    def process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and potentially transform metadata.
        
        Args:
            metadata: The metadata to process
            
        Returns:
            Processed metadata
        """
        ...


T = TypeVar('T')

class MetadataProvider(Generic[T], ABC):
    """
    Interface for metadata providers in Limnos.
    
    This interface defines the standard methods that all metadata providers
    must implement to ensure consistent metadata handling across frameworks.
    """
    
    @abstractmethod
    def get_metadata(self, document_id: str, format: MetadataFormat = MetadataFormat.DICT) -> T:
        """
        Get metadata for a document in the specified format.
        
        Args:
            document_id: ID of the document
            format: Format to return the metadata in
            
        Returns:
            Metadata in the requested format
        """
        pass
    
    @abstractmethod
    def get_metadata_schema(self) -> Dict[str, Any]:
        """
        Get the metadata schema definition.
        
        Returns:
            Dictionary representing the metadata schema
        """
        pass
    
    @abstractmethod
    def register_extension_point(self, name: str, extension: MetadataExtensionPoint) -> None:
        """
        Register an extension point for metadata processing.
        
        Args:
            name: Name of the extension point
            extension: Extension point implementation
        """
        pass
    
    @abstractmethod
    def get_extension_point(self, name: str) -> Optional[MetadataExtensionPoint]:
        """
        Get a registered extension point by name.
        
        Args:
            name: Name of the extension point
            
        Returns:
            Extension point implementation or None if not found
        """
        pass
    
    @abstractmethod
    def list_extension_points(self) -> List[str]:
        """
        List all registered extension points.
        
        Returns:
            List of extension point names
        """
        pass
    
    @abstractmethod
    def process_metadata_with_extension(self, 
                                       document_id: str, 
                                       extension_name: str) -> Dict[str, Any]:
        """
        Process metadata with a specific extension point.
        
        Args:
            document_id: ID of the document
            extension_name: Name of the extension point to use
            
        Returns:
            Processed metadata
        """
        pass


class UniversalMetadataProvider(MetadataProvider[Union[Dict[str, Any], UniversalMetadataSchema, str]]):
    """
    Implementation of the MetadataProvider interface for universal metadata.
    
    This class provides access to universal metadata stored by the document collector,
    along with extension points for framework-specific preprocessing.
    """
    
    def __init__(self, source_dir: Path):
        """
        Initialize the universal metadata provider.
        
        Args:
            source_dir: Directory containing universal metadata
        """
        self.source_dir = source_dir
        self.extension_points: Dict[str, MetadataExtensionPoint] = {}
        
    def get_metadata(self, 
                    document_id: str, 
                    format: MetadataFormat = MetadataFormat.DICT
                    ) -> Union[Dict[str, Any], UniversalMetadataSchema, str]:
        """
        Get metadata for a document in the specified format.
        
        Args:
            document_id: ID of the document
            format: Format to return the metadata in
            
        Returns:
            Metadata in the requested format
        """
        # Construct path to metadata file
        metadata_path = self.source_dir / document_id / f"{document_id}.json"
        
        # Check if metadata file exists
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found for document {document_id}")
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
        
        # Return in requested format
        if format == MetadataFormat.DICT:
            return metadata_dict
        elif format == MetadataFormat.SCHEMA:
            return UniversalMetadataSchema.from_dict(metadata_dict)
        elif format == MetadataFormat.JSON:
            return json.dumps(metadata_dict, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_metadata_schema(self) -> Dict[str, Any]:
        """
        Get the metadata schema definition.
        
        Returns:
            Dictionary representing the metadata schema
        """
        # Create a schema instance with default values
        schema = UniversalMetadataSchema(doc_id="example")
        
        # Convert to dictionary and return
        return schema.to_dict()
    
    def register_extension_point(self, name: str, extension: MetadataExtensionPoint) -> None:
        """
        Register an extension point for metadata processing.
        
        Args:
            name: Name of the extension point
            extension: Extension point implementation
        """
        self.extension_points[name] = extension
    
    def get_extension_point(self, name: str) -> Optional[MetadataExtensionPoint]:
        """
        Get a registered extension point by name.
        
        Args:
            name: Name of the extension point
            
        Returns:
            Extension point implementation or None if not found
        """
        return self.extension_points.get(name)
    
    def list_extension_points(self) -> List[str]:
        """
        List all registered extension points.
        
        Returns:
            List of extension point names
        """
        return list(self.extension_points.keys())
    
    def process_metadata_with_extension(self, 
                                       document_id: str, 
                                       extension_name: str) -> Dict[str, Any]:
        """
        Process metadata with a specific extension point.
        
        Args:
            document_id: ID of the document
            extension_name: Name of the extension point to use
            
        Returns:
            Processed metadata
        """
        # Get the extension point
        extension = self.get_extension_point(extension_name)
        if not extension:
            raise ValueError(f"Extension point not found: {extension_name}")
        
        # Get the metadata
        metadata = self.get_metadata(document_id, MetadataFormat.DICT)
        
        # Process with extension
        return extension.process_metadata(metadata)
    
    def bulk_process_metadata(self, 
                             document_ids: List[str], 
                             extension_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Process metadata for multiple documents with a specific extension point.
        
        Args:
            document_ids: List of document IDs
            extension_name: Name of the extension point to use
            
        Returns:
            Dictionary mapping document IDs to processed metadata
        """
        results = {}
        for doc_id in document_ids:
            try:
                results[doc_id] = self.process_metadata_with_extension(doc_id, extension_name)
            except Exception as e:
                # Log error but continue processing
                results[doc_id] = {"error": str(e)}
        
        return results
    
    def get_document_ids(self) -> List[str]:
        """
        Get a list of all document IDs with available metadata.
        
        Returns:
            List of document IDs
        """
        document_ids = []
        for item in self.source_dir.iterdir():
            if item.is_dir():
                metadata_file = item / f"{item.name}.json"
                if metadata_file.exists():
                    document_ids.append(item.name)
        
        return document_ids
