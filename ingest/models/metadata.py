"""
Universal Metadata Models for Limnos documents.

This module defines Pydantic models for metadata in the Limnos framework,
ensuring consistency, validation, and interoperability across different RAG frameworks.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, validator, root_validator


class DocumentType(str, Enum):
    """Enum defining the types of documents supported in Limnos."""
    
    ACADEMIC_PAPER = "academic_paper"
    DOCUMENTATION = "documentation"
    BOOK = "book"
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    CODE = "code"
    TEXT = "text"
    OTHER = "other"


class UniversalMetadata(BaseModel):
    """Universal metadata model for all documents in Limnos.
    
    This model defines the standard fields that should be present in all document
    metadata, regardless of the source or type of the document. Framework-specific
    metadata can extend this model.
    """
    
    # Identification
    doc_id: str
    doc_type: DocumentType = DocumentType.OTHER
    
    # Basic information
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    date_created: Optional[datetime] = None
    date_modified: Optional[datetime] = None
    collection_timestamp: datetime = Field(default_factory=datetime.now)
    
    # Content information
    language: str = "en"
    content_preview: Optional[str] = None
    content_length: Optional[int] = None
    summary: Optional[str] = None
    
    # Source information
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    storage_path: Optional[str] = None
    metadata_path: Optional[str] = None
    
    # File information
    filename: Optional[str] = None
    extension: Optional[str] = None
    size_bytes: Optional[int] = None
    num_pages: Optional[int] = None
    last_modified: Optional[datetime] = None
    
    # Semantic information
    keywords: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    
    # Structure information (document-specific)
    structure: Dict[str, bool] = Field(default_factory=dict)
    sections: List[Dict[str, str]] = Field(default_factory=list)
    
    # Additional fields specific to document types
    # Academic papers
    abstract: Optional[str] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    conference: Optional[str] = None
    references: List[str] = Field(default_factory=list)
    
    # For documentation
    version: Optional[str] = None
    api_version: Optional[str] = None
    framework: Optional[str] = None
    
    # Custom fields (for extensibility)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        arbitrary_types_allowed = True
        allow_population_by_field_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }
        
    @validator('doc_type', pre=True)
    def validate_doc_type(cls, v):
        """Validate and convert doc_type if needed."""
        if isinstance(v, str):
            try:
                return DocumentType(v)
            except ValueError:
                return DocumentType.OTHER
        return v
    
    @root_validator(pre=True)
    def handle_custom_fields(cls, values):
        """Handle custom fields that are not part of the model definition."""
        model_fields = cls.__fields__.keys()
        custom_fields = {k: v for k, v in values.items() if k not in model_fields}
        defined_fields = {k: v for k, v in values.items() if k in model_fields}
        
        # Add custom fields to the custom_fields dict
        if custom_fields:
            defined_fields['custom_fields'] = custom_fields
            
        return defined_fields
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a flat dictionary.
        
        Returns:
            Dictionary representation of the metadata with custom fields flattened
        """
        # Get the model as a dict
        data = self.dict(exclude_none=True)
        
        # Handle datetime serialization
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
                
        # Flatten custom fields into the main dictionary
        if 'custom_fields' in data and data['custom_fields']:
            custom = data.pop('custom_fields')
            data.update(custom)
            
        return data

    @classmethod
    def from_dict(cls, metadata_dict: Dict[str, Any]) -> 'UniversalMetadata':
        """Create a metadata model instance from a dictionary.
        
        Args:
            metadata_dict: Dictionary containing metadata
            
        Returns:
            UniversalMetadata instance
        """
        # The Pydantic constructor will handle validation and type conversion
        return cls(**metadata_dict)


class GraphRAGMetadata(BaseModel):
    """GraphRAG-specific metadata for documents.
    
    This model extends the universal metadata with GraphRAG-specific fields.
    """
    
    # Framework identification
    framework: str = "graphrag"
    
    # Configuration for entity extraction
    entity_extraction_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Configuration for relationship extraction
    relationship_extraction_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Configuration for graph construction
    graph_construction_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing status
    processing_status: Dict[str, Any] = Field(default_factory=dict)
    
    # Framework-specific custom fields
    graphrag_custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        arbitrary_types_allowed = True
        allow_population_by_field_name = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a flat dictionary."""
        data = self.dict(exclude_none=True)
        
        # Flatten custom fields
        if 'graphrag_custom_fields' in data and data['graphrag_custom_fields']:
            custom = data.pop('graphrag_custom_fields')
            data.update(custom)
            
        return data


class Section(BaseModel):
    """Model representing a section in a document.
    
    This model is used primarily for academic papers and other structured documents
    that have clear section divisions.
    """
    
    heading: str
    content: str
    level: int = 1
    section_id: Optional[str] = None
    parent_id: Optional[str] = None
    subsections: List['Section'] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        allow_population_by_field_name = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            'heading': self.heading,
            'content': self.content,
            'level': self.level,
            'section_id': self.section_id,
            'parent_id': self.parent_id,
            'metadata': self.metadata
        }
        
        if self.subsections:
            result['subsections'] = [s.to_dict() for s in self.subsections]
        
        return result


class PathRAGMetadata(BaseModel):
    """PathRAG-specific metadata for documents.
    
    This model extends the universal metadata with PathRAG-specific fields.
    """
    
    # Framework identification
    framework: str = "pathrag"
    
    # Configuration for chunking
    chunking_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Configuration for embedding
    embedding_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Configuration for path extraction
    path_extraction_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing status
    processing_status: Dict[str, Any] = Field(default_factory=dict)
    
    # Framework-specific custom fields
    pathrag_custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        arbitrary_types_allowed = True
        allow_population_by_field_name = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a flat dictionary."""
        data = self.dict(exclude_none=True)
        
        # Flatten custom fields
        if 'pathrag_custom_fields' in data and data['pathrag_custom_fields']:
            custom = data.pop('pathrag_custom_fields')
            data.update(custom)
            
        return data


class ProcessingStatusModel(BaseModel):
    """Model for tracking document processing status.
    
    This model provides a standard structure for tracking the processing status
    of documents across different frameworks.
    """
    
    # Basic processing steps
    collected: bool = False
    metadata_extracted: bool = False
    content_processed: bool = False
    
    # Framework-specific processing steps
    framework_processing: Dict[str, Dict[str, bool]] = Field(default_factory=dict)
    
    # Timestamps
    last_updated: datetime = Field(default_factory=datetime.now)
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }
    
    def update_status(self, step: str, status: bool) -> None:
        """Update a processing step status.
        
        Args:
            step: Name of the processing step
            status: New status (True = completed, False = not completed)
        """
        if hasattr(self, step):
            setattr(self, step, status)
        else:
            # For custom steps, add to framework_processing with default framework
            self.update_framework_status("universal", step, status)
            
        self.last_updated = datetime.now()
    
    def update_framework_status(self, framework: str, step: str, status: bool) -> None:
        """Update a framework-specific processing step status.
        
        Args:
            framework: Name of the framework
            step: Name of the processing step
            status: New status (True = completed, False = not completed)
        """
        if framework not in self.framework_processing:
            self.framework_processing[framework] = {}
            
        self.framework_processing[framework][step] = status
        self.last_updated = datetime.now()
    
    def is_fully_processed(self, framework: Optional[str] = None) -> bool:
        """Check if a document is fully processed.
        
        Args:
            framework: Optional framework name to check framework-specific steps
            
        Returns:
            True if fully processed, False otherwise
        """
        # Check basic processing
        if not (self.collected and self.metadata_extracted and self.content_processed):
            return False
            
        # If framework specified, check framework-specific steps
        if framework and framework in self.framework_processing:
            return all(self.framework_processing[framework].values())
            
        return True


def validate_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Validate metadata against the universal schema.
    
    Args:
        metadata: Metadata dictionary to validate
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    try:
        # Use Pydantic validation
        UniversalMetadata(**metadata)
    except Exception as e:
        # Convert Pydantic validation errors to a list of error messages
        if hasattr(e, 'errors'):
            for error in e.errors():
                errors.append(f"{error['loc']}: {error['msg']}")
        else:
            errors.append(str(e))
    
    return errors
