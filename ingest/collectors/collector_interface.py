"""
Interfaces for document collectors in Limnos.

This module defines simplified interfaces for document collectors
to avoid conflicts with the existing interface hierarchy.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from limnos.ingest.interface import Document


class CollectorComponent(ABC):
    """Base interface for collector components."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component with configuration.
        
        Args:
            config: Configuration dictionary for the component
        """
        pass
    
    @property
    @abstractmethod
    def component_type(self) -> str:
        """Return the type of this component."""
        pass
    
    @property
    @abstractmethod
    def component_name(self) -> str:
        """Return the name of this component."""
        pass


class DocumentCollector(CollectorComponent):
    """Interface for document collectors."""
    
    @abstractmethod
    def collect_file(self, file_path: Path) -> Tuple[Document, Dict[str, Path]]:
        """Collect a single file, process it, and store with metadata.
        
        Args:
            file_path: Path to the file to collect
            
        Returns:
            Tuple of (document, file_paths) where file_paths is a dict with paths to:
            - original: Path to the stored original document
            - metadata: Path to the stored metadata JSON file
        """
        pass
    
    @abstractmethod
    def collect_directory(self, directory_path: Path, recursive: bool = True,
                         file_extensions: Optional[List[str]] = None) -> List[Tuple[Document, Dict[str, Path]]]:
        """Collect all files in a directory, process them, and store with metadata.
        
        Args:
            directory_path: Path to the directory to collect
            recursive: Whether to process subdirectories
            file_extensions: Optional list of file extensions to process
            
        Returns:
            List of tuples containing (document, file_paths) for each collected document
        """
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Tuple[Document, Dict[str, Path]]]:
        """Retrieve a previously collected document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Tuple of (document, file_paths) if found, None otherwise
        """
        pass
    
    @abstractmethod
    def list_documents(self) -> List[str]:
        """List all collected documents.
        
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete a collected document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if deleted, False otherwise
        """
        pass
