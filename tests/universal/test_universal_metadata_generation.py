"""
Universal Metadata Generation Tests.

This module tests the generation of universal metadata from different document types,
ensuring all required fields are present and correctly formatted.
"""

import os
import unittest
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# We'll use pydantic for the improved test structure
from pydantic import BaseModel, Field, ValidationError
from datetime import datetime
from enum import Enum

# Import document processor for testing metadata generation
from limnos.ingest.processors.document_processor import DocumentProcessor
from limnos.ingest.interface import Document
from limnos.ingest.collectors.metadata_schema import DocumentType


# Define a Pydantic model for metadata validation
# This is aligned with our planned refactoring
class UniversalMetadataModel(BaseModel):
    """Pydantic model for universal metadata validation."""
    doc_id: str
    title: str
    doc_type: str
    content_preview: Optional[str] = None
    storage_path: Optional[str] = None
    metadata_path: Optional[str] = None
    collection_timestamp: Optional[str] = None
    size_bytes: Optional[int] = None
    num_pages: Optional[int] = None
    authors: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    
    class Config:
        """Pydantic config."""
        extra = "allow"  # Allow additional fields


class UniversalMetadataGenerationTest(unittest.TestCase):
    """Test case for universal metadata generation."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample files for testing
        self.sample_files = self._create_sample_files()
        
        # Initialize the processor
        self.processor = DocumentProcessor()
        self.processor.initialize({})
    
    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_files(self) -> Dict[str, Path]:
        """Create sample files for testing."""
        files = {}
        
        # Create a sample PDF file (with fake PDF header/footer)
        pdf_path = Path(self.temp_dir) / "sample_paper.pdf"
        with open(pdf_path, 'w', encoding='utf-8') as f:
            f.write("%PDF-1.5\n")
            f.write("Title: Sample Academic Paper\n")
            f.write("Authors: John Doe, Jane Smith\n\n")
            f.write("Abstract: This is a sample academic paper for testing metadata extraction.\n")
            f.write("Keywords: metadata, testing, academic\n\n")
            f.write("1. Introduction\n")
            f.write("This paper discusses the importance of metadata in academic research.\n")
            f.write("Multiple pages of content would follow in a real paper.\n")
            f.write("%EOF")
        files['pdf_paper'] = pdf_path
        
        # Create a sample text file with structured content
        txt_path = Path(self.temp_dir) / "sample_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("Sample Text Report\n")
            f.write("Author: Text Writer\n\n")
            f.write("This is a sample text report for testing metadata extraction.\n")
            f.write("It contains multiple paragraphs and sections.\n\n")
            f.write("Section 1: Introduction\n")
            f.write("The introduction explains the purpose of this report.\n")
        files['txt_report'] = txt_path
        
        # Create a sample markdown file
        md_path = Path(self.temp_dir) / "sample_documentation.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Sample Documentation\n\n")
            f.write("Author: Documentation Writer\n\n")
            f.write("This is sample documentation in markdown format.\n\n")
            f.write("## Section 1: Overview\n\n")
            f.write("The overview provides a high-level explanation.\n\n")
            f.write("## Section 2: Details\n\n")
            f.write("The details section provides more specific information.\n")
        files['md_doc'] = md_path
        
        return files
    
    def _validate_universal_metadata(self, metadata: Dict[str, Any]) -> None:
        """Validate that metadata conforms to universal schema."""
        try:
            # Use Pydantic for validation
            validated_metadata = UniversalMetadataModel(**metadata)
            
            # Extra checks beyond Pydantic validation
            self.assertIsNotNone(metadata.get('doc_id'))
            self.assertIsNotNone(metadata.get('title'))
            self.assertIsNotNone(metadata.get('doc_type'))
            
            # Check that doc_type is a valid DocumentType
            self.assertIn(metadata.get('doc_type'), 
                         [dt.value for dt in DocumentType])
            
        except ValidationError as e:
            self.fail(f"Metadata validation failed: {e}")
    
    def test_pdf_metadata_generation(self):
        """Test metadata generation from PDF files."""
        # Process a PDF file
        document = self.processor.process_file(self.sample_files['pdf_paper'])
        
        # Validate the generated metadata
        self._validate_universal_metadata(document.metadata)
        
        # Additional checks specific to PDF processing
        metadata = document.metadata
        self.assertEqual(metadata.get('doc_type'), DocumentType.ACADEMIC_PAPER.value)
        self.assertIn('num_pages', metadata)
        self.assertIn('Sample Academic Paper', metadata.get('title', ''))
        
        # Verify content extraction
        self.assertIn("sample academic paper", document.content.lower())
    
    def test_text_metadata_generation(self):
        """Test metadata generation from text files."""
        # Process a text file
        document = self.processor.process_file(self.sample_files['txt_report'])
        
        # Validate the generated metadata
        self._validate_universal_metadata(document.metadata)
        
        # Additional checks specific to text processing
        metadata = document.metadata
        self.assertEqual(metadata.get('doc_type'), DocumentType.TEXT.value)
        self.assertIn('Sample Text Report', metadata.get('title', ''))
        
        # Verify content extraction
        self.assertEqual(document.content, open(self.sample_files['txt_report'], 'r').read())
    
    def test_markdown_metadata_generation(self):
        """Test metadata generation from markdown files."""
        # Process a markdown file
        document = self.processor.process_file(self.sample_files['md_doc'])
        
        # Validate the generated metadata
        self._validate_universal_metadata(document.metadata)
        
        # Additional checks specific to markdown processing
        metadata = document.metadata
        self.assertEqual(metadata.get('doc_type'), DocumentType.TEXT.value)
        self.assertIn('Sample Documentation', metadata.get('title', ''))
        
        # Verify content extraction and header parsing
        self.assertIn("Sample Documentation", document.content)
        self.assertIn("Section 1: Overview", document.content)
    
    def test_metadata_field_consistency(self):
        """Test that all document types generate consistent metadata fields."""
        # Process all file types
        documents = {
            'pdf': self.processor.process_file(self.sample_files['pdf_paper']),
            'txt': self.processor.process_file(self.sample_files['txt_report']),
            'md': self.processor.process_file(self.sample_files['md_doc'])
        }
        
        # Get common fields across all document types
        common_fields = set.intersection(
            *[set(doc.metadata.keys()) for doc in documents.values()]
        )
        
        # Essential fields that must be present for all document types
        essential_fields = {'doc_id', 'title', 'doc_type', 'content'}
        
        # Check that all essential fields are in the common fields
        for field in essential_fields:
            self.assertIn(field, common_fields, 
                         f"Essential field '{field}' missing from common metadata fields")
    
    def test_metadata_storage_path_generation(self):
        """Test that metadata includes proper storage paths."""
        # Process a document
        document = self.processor.process_file(self.sample_files['pdf_paper'])
        metadata = document.metadata
        
        # Check storage path
        self.assertIn('storage_path', metadata)
        if 'storage_path' in metadata:
            self.assertTrue(Path(metadata['storage_path']).name.endswith('.pdf'))
        
        # Check metadata path
        self.assertIn('metadata_path', metadata)
        if 'metadata_path' in metadata:
            self.assertTrue(Path(metadata['metadata_path']).name.endswith('.json'))


if __name__ == '__main__':
    unittest.main()
