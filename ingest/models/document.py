"""
Document models for the Limnos framework.

This module defines Pydantic models for documents in the Limnos framework,
providing robust validation, serialization, and a clean interface for
working with document data.
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
import uuid

from pydantic import BaseModel, Field, validator

from limnos.ingest.models.metadata import UniversalMetadata


class Document(BaseModel):
    """Document model representing a document in the Limnos framework.
    
    This model provides a standardized representation of documents with
    content and metadata, with validation and serialization capabilities.
    """
    
    # Basic document properties
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    
    # Metadata can be a dict or a UniversalMetadata instance
    metadata: Union[Dict[str, Any], UniversalMetadata] = Field(default_factory=dict)
    
    # Optional file information
    file_path: Optional[Path] = None
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            Path: lambda v: str(v) if v else None,
        }
    
    @validator('metadata', pre=True)
    def validate_metadata(cls, v):
        """Ensure metadata is either a dict or UniversalMetadata."""
        if isinstance(v, dict):
            return v
        elif isinstance(v, UniversalMetadata):
            return v
        else:
            return dict(v)
    
    def ensure_universal_metadata(self) -> UniversalMetadata:
        """Ensure metadata is a UniversalMetadata instance.
        
        Returns:
            UniversalMetadata instance
        """
        if isinstance(self.metadata, dict):
            # Convert dict to UniversalMetadata
            if 'doc_id' not in self.metadata:
                self.metadata['doc_id'] = self.doc_id
            metadata = UniversalMetadata(**self.metadata)
            self.metadata = metadata
        return self.metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the document to a dictionary.
        
        Returns:
            Dictionary representation of the document
        """
        data = self.dict(exclude_none=True)
        
        # Handle metadata conversion
        if isinstance(self.metadata, UniversalMetadata):
            data['metadata'] = self.metadata.to_dict()
        
        # Handle Path conversion
        if self.file_path:
            data['file_path'] = str(self.file_path)
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create a document from a dictionary.
        
        Args:
            data: Dictionary containing document data
            
        Returns:
            Document instance
        """
        # Handle metadata conversion if it's a dict
        if 'metadata' in data and isinstance(data['metadata'], dict):
            # Keep it as a dict for now, the validator will handle it
            pass
            
        # Handle file_path conversion
        if 'file_path' in data and isinstance(data['file_path'], str):
            data['file_path'] = Path(data['file_path'])
            
        return cls(**data)


class DocumentChunk(BaseModel):
    """Document chunk model representing a portion of a document.
    
    This model provides a standardized representation of document chunks
    with content and metadata, with validation and serialization capabilities.
    """
    
    # Basic chunk properties
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    content: str = ""
    
    # Chunk metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Chunking information
    start_char_idx: Optional[int] = None
    end_char_idx: Optional[int] = None
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the chunk to a dictionary.
        
        Returns:
            Dictionary representation of the chunk
        """
        return self.dict(exclude_none=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create a chunk from a dictionary.
        
        Args:
            data: Dictionary containing chunk data
            
        Returns:
            DocumentChunk instance
        """
        return cls(**data)


class EmbeddedChunk(DocumentChunk):
    """Embedded chunk model representing a document chunk with embedding.
    
    This model extends DocumentChunk with embedding information.
    """
    
    # Embedding vector as a list of floats
    embedding: List[float] = Field(default_factory=list)
    
    # Embedding metadata
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    embedding_timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }
    
    @validator('embedding', pre=True)
    def validate_embedding(cls, v):
        """Ensure embedding is a list of floats."""
        # Convert numpy array to list if needed
        if hasattr(v, 'tolist'):
            return v.tolist()
        return v
