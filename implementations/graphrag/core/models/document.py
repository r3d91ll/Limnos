"""
Document Reference Model for GraphRAG

This module defines the DocumentReference class, which represents document metadata
in the GraphRAG implementation, following the Limnos architecture principles for
metadata separation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import os
import json


@dataclass
class DocumentReference:
    """
    Represents a reference to a document in the GraphRAG knowledge graph.
    
    DocumentReference objects maintain links between graph elements and their
    source documents, following the Limnos architecture principles for metadata
    separation between universal and framework-specific metadata.
    """
    
    # Core attributes
    document_id: str
    title: str
    
    # Document metadata
    document_type: str = "text"
    source_path: Optional[str] = None
    metadata_path: Optional[str] = None
    
    # Document properties
    publication_date: Optional[datetime] = None
    authors: List[str] = field(default_factory=list)
    
    # GraphRAG-specific metadata
    processed: bool = False
    last_processed: Optional[datetime] = None
    entity_count: int = 0
    relationship_count: int = 0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def universal_metadata_path(self) -> Optional[str]:
        """
        Get the path to the universal metadata file for this document.
        
        Following Limnos architecture principles, universal metadata is stored
        alongside source documents with matching filenames but different extensions.
        
        Returns:
            Path to the universal metadata file, or None if source_path is not set
        """
        if not self.source_path:
            return None
            
        base_path, _ = os.path.splitext(self.source_path)
        return f"{base_path}.json"
    
    @property
    def graphrag_metadata_path(self) -> Optional[str]:
        """
        Get the path to the GraphRAG-specific metadata file for this document.
        
        Following Limnos architecture principles, framework-specific metadata is
        stored in separate directories specific to each RAG implementation.
        
        Returns:
            Path to the GraphRAG-specific metadata file, or None if source_path is not set
        """
        if not self.source_path:
            return None
            
        # Extract filename from source path
        filename = os.path.basename(self.source_path)
        base_filename, _ = os.path.splitext(filename)
        
        # Construct path to GraphRAG-specific metadata
        return os.path.join(
            "/home/todd/ML-Lab/Olympus/limnos/data/implementations/graphrag/documents",
            f"{base_filename}.json"
        )
    
    def load_universal_metadata(self) -> Dict[str, Any]:
        """
        Load universal metadata for this document.
        
        Returns:
            Dictionary containing universal metadata, or empty dict if not found
        """
        metadata_path = self.universal_metadata_path
        if not metadata_path or not os.path.exists(metadata_path):
            return {}
            
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def save_graphrag_metadata(self) -> bool:
        """
        Save GraphRAG-specific metadata for this document.
        
        Returns:
            True if successful, False otherwise
        """
        metadata_path = self.graphrag_metadata_path
        if not metadata_path:
            return False
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            return True
        except IOError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the document reference to a dictionary representation.
        
        Returns:
            Dictionary representation of the document reference
        """
        result = {
            "document_id": self.document_id,
            "title": self.title,
            "document_type": self.document_type,
            "processed": self.processed,
            "entity_count": self.entity_count,
            "relationship_count": self.relationship_count,
            "metadata": self.metadata
        }
        
        # Add optional fields if they exist
        if self.source_path:
            result["source_path"] = self.source_path
            
        if self.metadata_path:
            result["metadata_path"] = self.metadata_path
            
        if self.publication_date:
            result["publication_date"] = self.publication_date.isoformat()
            
        if self.authors:
            result["authors"] = self.authors
            
        if self.last_processed:
            result["last_processed"] = self.last_processed.isoformat()
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentReference':
        """
        Create a DocumentReference instance from a dictionary.
        
        Args:
            data: Dictionary representation of a document reference
            
        Returns:
            DocumentReference instance
        """
        # Create a copy to avoid modifying the input
        doc_data = data.copy()
        
        # Handle date fields
        if "publication_date" in doc_data and isinstance(doc_data["publication_date"], str):
            try:
                doc_data["publication_date"] = datetime.fromisoformat(doc_data["publication_date"])
            except ValueError:
                doc_data["publication_date"] = None
                
        if "last_processed" in doc_data and isinstance(doc_data["last_processed"], str):
            try:
                doc_data["last_processed"] = datetime.fromisoformat(doc_data["last_processed"])
            except ValueError:
                doc_data["last_processed"] = None
        
        # Create the document reference
        return cls(**doc_data)
    
    @classmethod
    def from_universal_metadata(cls, document_id: str, source_path: str) -> 'DocumentReference':
        """
        Create a DocumentReference from universal metadata.
        
        Args:
            document_id: Unique identifier for the document
            source_path: Path to the source document
            
        Returns:
            DocumentReference instance populated from universal metadata
        """
        doc_ref = cls(document_id=document_id, title="", source_path=source_path)
        
        # Load universal metadata
        metadata = doc_ref.load_universal_metadata()
        if not metadata:
            # If no metadata found, use basic information
            base_filename = os.path.basename(source_path)
            title, _ = os.path.splitext(base_filename)
            doc_ref.title = title
            return doc_ref
            
        # Populate from metadata
        doc_ref.title = metadata.get("title", "")
        doc_ref.document_type = metadata.get("document_type", "text")
        
        # Handle publication date
        pub_date = metadata.get("publication_date")
        if pub_date and isinstance(pub_date, str):
            try:
                doc_ref.publication_date = datetime.fromisoformat(pub_date)
            except ValueError:
                pass
                
        # Handle authors
        authors = metadata.get("authors", [])
        if isinstance(authors, list):
            doc_ref.authors = authors
            
        # Store remaining metadata
        for key, value in metadata.items():
            if key not in ["title", "document_type", "publication_date", "authors"]:
                doc_ref.metadata[key] = value
                
        return doc_ref
