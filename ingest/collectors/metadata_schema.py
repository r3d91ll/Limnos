"""
Universal Metadata Schema for Limnos documents.

This module defines the standard metadata schema for all documents in Limnos,
ensuring consistency and interoperability across different RAG frameworks.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field


class DocumentType(Enum):
    """Enum defining the types of documents supported in Limnos."""
    
    ACADEMIC_PAPER = "academic_paper"
    DOCUMENTATION = "documentation"
    BOOK = "book"
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    CODE = "code"
    OTHER = "other"


@dataclass
class UniversalMetadataSchema:
    """Universal metadata schema for all documents in Limnos.
    
    This schema defines the standard fields that should be present in all document
    metadata, regardless of the source or type of the document. Framework-specific
    metadata can extend this schema.
    """
    
    # Identification
    doc_id: str
    doc_type: DocumentType = DocumentType.OTHER
    
    # Basic information
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    date_created: Optional[str] = None
    date_modified: Optional[str] = None
    
    # Content information
    language: str = "en"
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
    last_modified: Optional[float] = None
    
    # Semantic information
    keywords: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    
    # Structure information (document-specific)
    structure: Dict[str, bool] = field(default_factory=dict)
    sections: List[Dict[str, str]] = field(default_factory=list)
    
    # Additional fields specific to document types
    # Academic papers
    abstract: Optional[str] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    conference: Optional[str] = None
    references: List[str] = field(default_factory=list)
    
    # For documentation
    version: Optional[str] = None
    api_version: Optional[str] = None
    framework: Optional[str] = None
    
    # Custom fields (for extensibility)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, metadata_dict: Dict[str, Any]) -> 'UniversalMetadataSchema':
        """Create a metadata schema instance from a dictionary.
        
        Args:
            metadata_dict: Dictionary containing metadata
            
        Returns:
            UniversalMetadataSchema instance
        """
        # Handle DocumentType enum
        if 'doc_type' in metadata_dict and isinstance(metadata_dict['doc_type'], str):
            try:
                metadata_dict['doc_type'] = DocumentType(metadata_dict['doc_type'])
            except ValueError:
                metadata_dict['doc_type'] = DocumentType.OTHER
        
        # Filter dictionary to only include fields defined in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in metadata_dict.items() if k in valid_fields}
        
        # Add any additional fields to custom
        custom_fields = {k: v for k, v in metadata_dict.items() if k not in valid_fields}
        if custom_fields:
            filtered_dict['custom'] = custom_fields
        
        return cls(**filtered_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the schema to a dictionary.
        
        Returns:
            Dictionary representation of the metadata
        """
        result = {}
        
        for field_name, field_value in self.__dict__.items():
            # Handle Enum conversion
            if isinstance(field_value, Enum):
                result[field_name] = field_value.value
            else:
                result[field_name] = field_value
        
        # Merge custom fields into the main dictionary for flat structure
        if 'custom' in result and result['custom']:
            custom = result.pop('custom')
            result.update(custom)
        
        return result


def validate_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Validate metadata against the universal schema.
    
    Args:
        metadata: Metadata dictionary to validate
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Check required fields
    if 'doc_id' not in metadata:
        errors.append("Missing required field: doc_id")
    
    # Validate field types
    if 'authors' in metadata and not isinstance(metadata['authors'], list):
        errors.append("Field 'authors' must be a list")
    
    if 'keywords' in metadata and not isinstance(metadata['keywords'], list):
        errors.append("Field 'keywords' must be a list")
    
    if 'sections' in metadata and not isinstance(metadata['sections'], list):
        errors.append("Field 'sections' must be a list")
    
    # Add more validation rules as needed
    
    return errors
