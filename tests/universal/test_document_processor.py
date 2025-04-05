"""
Document Processor Tests.

This module provides comprehensive tests for the Document Processor,
which handles the extraction of content and metadata from documents.
"""

import os
import unittest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List

from limnos.ingest.processors.document_processor import DocumentProcessor
from limnos.ingest.processors.pdf_processor import PDFProcessor
from limnos.ingest.processors.text_processor import TextProcessor
from limnos.ingest.interface import Document
from limnos.ingest.collectors.metadata_schema import DocumentType


class DocumentProcessorTest(unittest.TestCase):
    """Test case for the Document Processor."""
    
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
        
        # Create a sample PDF file (actually just a text file with .pdf extension)
        pdf_path = Path(self.temp_dir) / "sample.pdf"
        with open(pdf_path, 'w', encoding='utf-8') as f:
            f.write("%PDF-1.4\nThis is a sample PDF file content.\nIt has multiple lines.\n%EOF")
        files['pdf'] = pdf_path
        
        # Create a sample text file
        txt_path = Path(self.temp_dir) / "sample.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("This is a sample text file content.\nIt has multiple lines.")
        files['txt'] = txt_path
        
        # Create a sample markdown file
        md_path = Path(self.temp_dir) / "sample.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Sample Markdown\n\nThis is a sample markdown file content.\nIt has multiple lines.")
        files['md'] = md_path
        
        return files
    
    def test_initialization(self):
        """Test processor initialization."""
        # Check that the processor is initialized
        self.assertTrue(self.processor._initialized)
        
        # Check registered processors
        processors = self.processor._processors
        self.assertIsInstance(processors.get('.pdf'), PDFProcessor)
        self.assertIsInstance(processors.get('.txt'), TextProcessor)
        self.assertIsInstance(processors.get('.md'), TextProcessor)
    
    def test_process_pdf(self):
        """Test processing a PDF file."""
        # Process a PDF file
        document = self.processor.process_file(self.sample_files['pdf'])
        
        # Check document
        self.assertIsInstance(document, Document)
        self.assertIsNotNone(document.doc_id)
        self.assertEqual(document.metadata.get('doc_type'), DocumentType.ACADEMIC_PAPER.value)
        self.assertIn('content', document.metadata)
        self.assertIn('title', document.metadata)
        self.assertIn('num_pages', document.metadata)
        
        # Check content extraction
        self.assertIn("sample PDF file content", document.content)
    
    def test_process_text(self):
        """Test processing a text file."""
        # Process a text file
        document = self.processor.process_file(self.sample_files['txt'])
        
        # Check document
        self.assertIsInstance(document, Document)
        self.assertIsNotNone(document.doc_id)
        self.assertEqual(document.metadata.get('doc_type'), DocumentType.TEXT.value)
        self.assertIn('content', document.metadata)
        self.assertIn('title', document.metadata)
        
        # Check content extraction
        self.assertEqual(document.content, "This is a sample text file content.\nIt has multiple lines.")
    
    def test_process_markdown(self):
        """Test processing a markdown file."""
        # Process a markdown file
        document = self.processor.process_file(self.sample_files['md'])
        
        # Check document
        self.assertIsInstance(document, Document)
        self.assertIsNotNone(document.doc_id)
        self.assertEqual(document.metadata.get('doc_type'), DocumentType.TEXT.value)
        self.assertIn('content', document.metadata)
        self.assertIn('title', document.metadata)
        
        # Check content extraction
        self.assertIn("Sample Markdown", document.content)
        self.assertIn("sample markdown file content", document.content)
    
    def test_unsupported_file_type(self):
        """Test processing an unsupported file type."""
        # Create an unsupported file
        unsupported_path = Path(self.temp_dir) / "sample.xyz"
        with open(unsupported_path, 'w', encoding='utf-8') as f:
            f.write("This is an unsupported file type.")
        
        # Process should raise an error
        with self.assertRaises(ValueError):
            self.processor.process_file(unsupported_path)
    
    def test_register_processor(self):
        """Test registering a custom processor."""
        # Create a mock processor
        class MockProcessor:
            def process(self, file_path: Path) -> Document:
                doc = Document(doc_id="mock_doc", content="Mock content")
                doc.metadata = {
                    'title': 'Mock Document',
                    'doc_type': 'mock',
                    'content': 'Mock content'
                }
                return doc
        
        # Register the mock processor
        self.processor.register_processor('.mock', MockProcessor())
        
        # Create a mock file
        mock_path = Path(self.temp_dir) / "sample.mock"
        with open(mock_path, 'w', encoding='utf-8') as f:
            f.write("This is a mock file.")
        
        # Process the mock file
        document = self.processor.process_file(mock_path)
        
        # Check document
        self.assertIsInstance(document, Document)
        self.assertEqual(document.doc_id, "mock_doc")
        self.assertEqual(document.metadata.get('doc_type'), 'mock')
        self.assertEqual(document.content, "Mock content")
    
    def test_get_supported_extensions(self):
        """Test getting supported file extensions."""
        # Get supported extensions
        extensions = self.processor.get_supported_extensions()
        
        # Check extensions
        self.assertIn('.pdf', extensions)
        self.assertIn('.txt', extensions)
        self.assertIn('.md', extensions)
        
        # Register a new processor and check again
        class MockProcessor:
            def process(self, file_path: Path) -> Document:
                return Document(doc_id="mock_doc", content="Mock content")
        
        self.processor.register_processor('.mock', MockProcessor())
        
        extensions = self.processor.get_supported_extensions()
        self.assertIn('.mock', extensions)


if __name__ == '__main__':
    unittest.main()
