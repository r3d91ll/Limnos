"""
Universal Document Collector Core Tests.

This module provides focused tests for the core functionality of the Universal Document Collector,
including document collection, storage, and retrieval.
"""

import os
import unittest
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from limnos.ingest.collectors.universal_collector import UniversalDocumentCollector
from limnos.ingest.interface import Document


class UniversalDocumentCollectorCoreTest(unittest.TestCase):
    """Test case for the core functionality of the Universal Document Collector."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.source_dir = Path(self.temp_dir) / "source_documents"
        os.makedirs(self.source_dir, exist_ok=True)
        
        # Create sample files for testing
        self.sample_files = self._create_sample_files()
        
        # Initialize the collector
        self.collector = UniversalDocumentCollector()
        self.collector.initialize({
            'source_dir': str(self.source_dir)
        })
    
    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_files(self) -> Dict[str, Path]:
        """Create sample files for testing."""
        files = {}
        
        # Create a sample PDF file (actually just a text file with .pdf extension)
        pdf_path = Path(self.temp_dir) / "sample.pdf"
        with open(pdf_path, 'w', encoding='utf-8') as f:
            f.write("%PDF-1.4\nThis is a sample PDF file content.\n%EOF")
        files['pdf'] = pdf_path
        
        # Create a sample text file
        txt_path = Path(self.temp_dir) / "sample.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("This is a sample text file content.")
        files['txt'] = txt_path
        
        # Create a sample markdown file
        md_path = Path(self.temp_dir) / "sample.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Sample Markdown\n\nThis is a sample markdown file content.")
        files['md'] = md_path
        
        # Create a directory with multiple files
        multi_dir = Path(self.temp_dir) / "multi"
        os.makedirs(multi_dir, exist_ok=True)
        
        for i in range(3):
            file_path = multi_dir / f"sample{i}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"This is sample file {i} content.")
        
        files['multi_dir'] = multi_dir
        
        return files
    
    def test_initialization(self):
        """Test collector initialization."""
        # Check that the collector is initialized
        self.assertTrue(self.collector._initialized)
        self.assertEqual(self.collector._source_dir, self.source_dir)
        
        # Check that the document index is empty (no documents collected yet)
        self.assertEqual(len(self.collector.get_document_index()), 0)
        
        # Check metadata provider
        metadata_provider = self.collector.get_metadata_provider()
        self.assertIsNotNone(metadata_provider)
    
    def test_collect_file(self):
        """Test collecting a single file."""
        # Collect a PDF file
        document, file_paths = self.collector.collect_file(self.sample_files['pdf'])
        
        # Check document
        self.assertIsInstance(document, Document)
        self.assertIsNotNone(document.metadata.get('doc_id'))
        
        # Check file paths
        self.assertIn('original', file_paths)
        self.assertIn('metadata', file_paths)
        self.assertTrue(os.path.exists(file_paths['original']))
        self.assertTrue(os.path.exists(file_paths['metadata']))
        
        # Check document index
        doc_id = document.metadata['doc_id']
        doc_index = self.collector.get_document_index()
        self.assertIn(doc_id, doc_index)
        
        # Check metadata file content
        with open(file_paths['metadata'], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        self.assertEqual(metadata['doc_id'], doc_id)
        self.assertIn('storage_path', metadata)
        self.assertIn('metadata_path', metadata)
        
        # Collect a text file
        document, file_paths = self.collector.collect_file(self.sample_files['txt'])
        
        # Check document
        self.assertIsInstance(document, Document)
        self.assertIsNotNone(document.metadata.get('doc_id'))
        
        # Check document index
        doc_id = document.metadata['doc_id']
        doc_index = self.collector.get_document_index()
        self.assertIn(doc_id, doc_index)
    
    def test_collect_directory(self):
        """Test collecting a directory of files."""
        # Collect the multi directory
        results = self.collector.collect_directory(self.sample_files['multi_dir'], recursive=True)
        
        # Check results
        self.assertEqual(len(results), 3)
        for document, file_paths in results:
            self.assertIsInstance(document, Document)
            self.assertIsNotNone(document.metadata.get('doc_id'))
            self.assertIn('original', file_paths)
            self.assertIn('metadata', file_paths)
            
        # Check document index
        doc_index = self.collector.get_document_index()
        self.assertEqual(len(doc_index), 3)
        
        # Test non-recursive collection
        # First clear the source directory
        shutil.rmtree(self.source_dir)
        os.makedirs(self.source_dir, exist_ok=True)
        self.collector.refresh_document_index()
        
        # Create a nested directory structure
        nested_dir = Path(self.temp_dir) / "nested"
        os.makedirs(nested_dir, exist_ok=True)
        os.makedirs(nested_dir / "subdir", exist_ok=True)
        
        # Create files in both directories
        with open(nested_dir / "file1.txt", 'w', encoding='utf-8') as f:
            f.write("File 1 content")
        with open(nested_dir / "subdir" / "file2.txt", 'w', encoding='utf-8') as f:
            f.write("File 2 content")
        
        # Collect without recursion
        results = self.collector.collect_directory(nested_dir, recursive=False, file_extensions=['.txt'])
        
        # Should only collect the top-level file
        self.assertEqual(len(results), 1)
        
        # Collect with recursion
        results = self.collector.collect_directory(nested_dir, recursive=True, file_extensions=['.txt'])
        
        # Should collect both files
        self.assertEqual(len(results), 2)
    
    def test_get_document(self):
        """Test retrieving a collected document."""
        # Collect a file
        original_document, file_paths = self.collector.collect_file(self.sample_files['txt'])
        doc_id = original_document.metadata['doc_id']
        
        # Retrieve the document
        retrieved = self.collector.get_document(doc_id)
        self.assertIsNotNone(retrieved)
        
        retrieved_document, retrieved_paths = retrieved
        self.assertEqual(retrieved_document.doc_id, doc_id)
        self.assertEqual(retrieved_paths['original'], file_paths['original'])
        self.assertEqual(retrieved_paths['metadata'], file_paths['metadata'])
        
        # Try retrieving a non-existent document
        self.assertIsNone(self.collector.get_document("non_existent_id"))
    
    def test_list_documents(self):
        """Test listing all collected documents."""
        # Initially, no documents
        self.assertEqual(len(self.collector.list_documents()), 0)
        
        # Collect some files
        self.collector.collect_file(self.sample_files['pdf'])
        self.collector.collect_file(self.sample_files['txt'])
        self.collector.collect_file(self.sample_files['md'])
        
        # Check list
        document_list = self.collector.list_documents()
        self.assertEqual(len(document_list), 3)
    
    def test_delete_document(self):
        """Test deleting a collected document."""
        # Collect a file
        document, _ = self.collector.collect_file(self.sample_files['txt'])
        doc_id = document.metadata['doc_id']
        
        # Verify it exists
        self.assertIn(doc_id, self.collector.list_documents())
        
        # Delete it
        result = self.collector.delete_document(doc_id)
        self.assertTrue(result)
        
        # Verify it's gone
        self.assertNotIn(doc_id, self.collector.list_documents())
        
        # Try deleting a non-existent document
        result = self.collector.delete_document("non_existent_id")
        self.assertFalse(result)
    
    def test_document_index(self):
        """Test document index management."""
        # Initially, index is empty
        self.assertEqual(len(self.collector.get_document_index()), 0)
        
        # Collect some files
        doc_ids = []
        for file_key in ['pdf', 'txt', 'md']:
            document, _ = self.collector.collect_file(self.sample_files[file_key])
            doc_ids.append(document.metadata['doc_id'])
        
        # Check index
        doc_index = self.collector.get_document_index()
        self.assertEqual(len(doc_index), 3)
        for doc_id in doc_ids:
            self.assertIn(doc_id, doc_index)
            self.assertIn('title', doc_index[doc_id])
            self.assertIn('doc_type', doc_index[doc_id])
            self.assertIn('collection_timestamp', doc_index[doc_id])
        
        # Delete a document and check index
        self.collector.delete_document(doc_ids[0])
        doc_index = self.collector.get_document_index()
        self.assertEqual(len(doc_index), 2)
        self.assertNotIn(doc_ids[0], doc_index)
        
        # Refresh index
        self.collector.refresh_document_index()
        doc_index = self.collector.get_document_index()
        self.assertEqual(len(doc_index), 2)


if __name__ == '__main__':
    unittest.main()
