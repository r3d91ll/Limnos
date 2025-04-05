"""
Universal Document Collector Tests.

This module provides comprehensive tests for the Universal Document Collector,
which serves as the foundation for all RAG frameworks in Limnos.
"""

import os
import unittest
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List

from limnos.ingest.collectors.universal_collector import UniversalDocumentCollector
from limnos.ingest.collectors.metadata_interface import MetadataExtensionPoint
from limnos.ingest.collectors.metadata_schema import DocumentType
from limnos.ingest.interface import Document


class MockMetadataExtensionPoint(MetadataExtensionPoint):
    """Mock implementation of a metadata extension point for testing."""
    
    def __init__(self, framework_name: str):
        """Initialize the mock extension point."""
        self.framework_name = framework_name
        self.processed_documents = {}
        
    def transform_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Transform universal metadata into framework-specific format."""
        # Make a copy and add framework-specific fields
        framework_metadata = metadata.copy()
        framework_metadata['framework'] = self.framework_name
        framework_metadata['processed_by'] = f"{self.framework_name}_processor"
        framework_metadata['framework_specific_field'] = f"value_for_{self.framework_name}"
        
        # Store the processed metadata
        doc_id = metadata.get('doc_id')
        if doc_id:
            self.processed_documents[doc_id] = framework_metadata
            
        return framework_metadata
    
    def get_framework_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get framework-specific metadata for a document."""
        return self.processed_documents.get(doc_id)
    
    def get_required_metadata_fields(self) -> set:
        """Get the set of required metadata fields."""
        return {'doc_id', 'title', 'content'}


class UniversalDocumentCollectorTest(unittest.TestCase):
    """Test case for the Universal Document Collector."""
    
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
        
        # Create mock extension points
        self.graphrag_extension = MockMetadataExtensionPoint('graphrag')
        self.pathrag_extension = MockMetadataExtensionPoint('pathrag')
        
        # Register frameworks and extension points
        self.collector.register_framework('graphrag')
        self.collector.register_framework('pathrag')
        self.collector.register_metadata_extension_point('graphrag', self.graphrag_extension)
        self.collector.register_metadata_extension_point('pathrag', self.pathrag_extension)
    
    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_files(self) -> Dict[str, Path]:
        """Create sample files for testing."""
        files = {}
        
        # Create a sample PDF file (actually just a text file with .pdf extension)
        pdf_path = Path(self.temp_dir) / "sample.pdf"
        with open(pdf_path, 'w', encoding='utf-8') as f:
            f.write("This is a sample PDF file content.")
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
        
        # Check registered frameworks
        self.assertEqual(self.collector.get_registered_frameworks(), {'graphrag', 'pathrag'})
        
        # Check metadata provider
        metadata_provider = self.collector.get_metadata_provider()
        self.assertIsNotNone(metadata_provider)
        
        # Check extension points
        extension_points = metadata_provider.list_extension_points()
        self.assertEqual(set(extension_points), {'graphrag', 'pathrag'})
    
    def test_collect_file(self):
        """Test collecting a single file."""
        # Collect a PDF file
        document, file_paths = self.collector.collect_file(self.sample_files['pdf'])
        
        # Check document
        self.assertIsInstance(document, Document)
        self.assertIsNotNone(document.metadata.get('doc_id'))
        self.assertEqual(document.metadata.get('doc_type'), DocumentType.ACADEMIC_PAPER.value)
        
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
        self.assertEqual(metadata['doc_type'], DocumentType.ACADEMIC_PAPER.value)
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
    
    def test_metadata_extension_points(self):
        """Test metadata extension points."""
        # Collect a file
        document, _ = self.collector.collect_file(self.sample_files['pdf'])
        doc_id = document.metadata['doc_id']
        
        # Process with GraphRAG extension
        graphrag_metadata = self.collector.process_metadata_with_extension(doc_id, 'graphrag')
        
        # Check GraphRAG-specific fields
        self.assertEqual(graphrag_metadata['framework'], 'graphrag')
        self.assertEqual(graphrag_metadata['processed_by'], 'graphrag_processor')
        self.assertEqual(graphrag_metadata['framework_specific_field'], 'value_for_graphrag')
        
        # Process with PathRAG extension
        pathrag_metadata = self.collector.process_metadata_with_extension(doc_id, 'pathrag')
        
        # Check PathRAG-specific fields
        self.assertEqual(pathrag_metadata['framework'], 'pathrag')
        self.assertEqual(pathrag_metadata['processed_by'], 'pathrag_processor')
        self.assertEqual(pathrag_metadata['framework_specific_field'], 'value_for_pathrag')
        
        # Get framework-specific metadata
        graphrag_metadata2 = self.collector.get_metadata_for_framework(doc_id, 'graphrag')
        self.assertEqual(graphrag_metadata, graphrag_metadata2)
        
        # Try with non-existent framework
        with self.assertRaises(ValueError):
            self.collector.get_metadata_for_framework(doc_id, 'non_existent_framework')
    
    def test_bulk_process_metadata(self):
        """Test bulk processing of metadata."""
        # Collect multiple files
        doc_ids = []
        for file_key in ['pdf', 'txt', 'md']:
            document, _ = self.collector.collect_file(self.sample_files[file_key])
            doc_ids.append(document.metadata['doc_id'])
        
        # Bulk process with GraphRAG
        results = self.collector.bulk_process_metadata(doc_ids, 'graphrag')
        
        # Check results
        self.assertEqual(len(results), 3)
        for doc_id in doc_ids:
            self.assertIn(doc_id, results)
            self.assertEqual(results[doc_id]['framework'], 'graphrag')
            
        # Try with non-existent framework
        with self.assertRaises(ValueError):
            self.collector.bulk_process_metadata(doc_ids, 'non_existent_framework')
    
    def test_metadata_provider(self):
        """Test the metadata provider interface."""
        # Collect a file
        document, _ = self.collector.collect_file(self.sample_files['pdf'])
        doc_id = document.metadata['doc_id']
        
        # Get metadata provider
        provider = self.collector.get_metadata_provider()
        
        # Get metadata in different formats
        dict_metadata = provider.get_metadata(doc_id, MetadataFormat.DICT)
        self.assertIsInstance(dict_metadata, dict)
        self.assertEqual(dict_metadata['doc_id'], doc_id)
        
        json_metadata = provider.get_metadata(doc_id, MetadataFormat.JSON)
        self.assertIsInstance(json_metadata, str)
        self.assertIn(doc_id, json_metadata)
        
        # Check extension points
        extension_points = provider.list_extension_points()
        self.assertEqual(set(extension_points), {'graphrag', 'pathrag'})
        
        graphrag_extension = provider.get_extension_point('graphrag')
        self.assertEqual(graphrag_extension, self.graphrag_extension)
        
        # Process with extension
        processed = provider.process_metadata_with_extension(doc_id, 'graphrag')
        self.assertEqual(processed['framework'], 'graphrag')


if __name__ == '__main__':
    unittest.main()
