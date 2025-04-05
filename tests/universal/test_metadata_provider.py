"""
Metadata Provider Tests.

This module provides comprehensive tests for the Metadata Provider interface,
which is a core universal component of the Limnos framework.
"""

import os
import unittest
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from limnos.ingest.collectors.metadata_interface import (
    MetadataProvider, 
    MetadataFormat, 
    MetadataExtensionPoint
)
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


class TestMetadataProvider(MetadataProvider):
    """Test implementation of the MetadataProvider interface."""
    
    def __init__(self, metadata_dir: Path):
        """Initialize the test metadata provider."""
        self.metadata_dir = metadata_dir
        self.extension_points = {}
        
    def get_metadata(self, document_id: str, format: MetadataFormat = MetadataFormat.DICT) -> Any:
        """Get metadata for a document."""
        metadata_path = self.metadata_dir / document_id / f"{document_id}.json"
        
        if not metadata_path.exists():
            return None
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        if format == MetadataFormat.DICT:
            return metadata
        elif format == MetadataFormat.JSON:
            return json.dumps(metadata, indent=2)
        else:
            raise ValueError(f"Unsupported metadata format: {format}")
    
    def register_extension_point(self, name: str, extension: MetadataExtensionPoint) -> None:
        """Register a metadata extension point."""
        self.extension_points[name] = extension
        
    def get_extension_point(self, name: str) -> Optional[MetadataExtensionPoint]:
        """Get a metadata extension point by name."""
        return self.extension_points.get(name)
        
    def list_extension_points(self) -> List[str]:
        """List all registered extension points."""
        return list(self.extension_points.keys())
        
    def process_metadata_with_extension(self, document_id: str, extension_name: str) -> Dict[str, Any]:
        """Process metadata with a specific extension point."""
        extension = self.get_extension_point(extension_name)
        if not extension:
            raise ValueError(f"Extension point not found: {extension_name}")
            
        metadata = self.get_metadata(document_id)
        if not metadata:
            raise ValueError(f"Metadata not found for document: {document_id}")
            
        return extension.transform_metadata(metadata)
        
    def bulk_process_metadata(self, document_ids: List[str], extension_name: str) -> Dict[str, Dict[str, Any]]:
        """Process metadata for multiple documents with a specific extension."""
        extension = self.get_extension_point(extension_name)
        if not extension:
            raise ValueError(f"Extension point not found: {extension_name}")
            
        results = {}
        for doc_id in document_ids:
            metadata = self.get_metadata(doc_id)
            if metadata:
                results[doc_id] = extension.transform_metadata(metadata)
                
        return results


class MetadataProviderTest(unittest.TestCase):
    """Test case for the Metadata Provider interface."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.metadata_dir = Path(self.temp_dir) / "metadata"
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Create sample metadata
        self.sample_docs = self._create_sample_metadata()
        
        # Initialize the provider
        self.provider = TestMetadataProvider(self.metadata_dir)
        
        # Create mock extension points
        self.graphrag_extension = MockMetadataExtensionPoint('graphrag')
        self.pathrag_extension = MockMetadataExtensionPoint('pathrag')
        
        # Register extension points
        self.provider.register_extension_point('graphrag', self.graphrag_extension)
        self.provider.register_extension_point('pathrag', self.pathrag_extension)
    
    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_metadata(self) -> Dict[str, str]:
        """Create sample metadata for testing."""
        docs = {}
        
        # Create sample documents with metadata
        for i in range(3):
            doc_id = f"doc_{i}"
            doc_dir = self.metadata_dir / doc_id
            os.makedirs(doc_dir, exist_ok=True)
            
            metadata = {
                'doc_id': doc_id,
                'title': f"Document {i}",
                'doc_type': DocumentType.ACADEMIC_PAPER.value,
                'content': f"Content of document {i}",
                'storage_path': str(doc_dir / f"{doc_id}.pdf"),
                'metadata_path': str(doc_dir / f"{doc_id}.json"),
                'collection_timestamp': '2025-04-05T07:00:00',
                'size_bytes': 1000 + i * 100,
                'num_pages': 10 + i,
                'authors': [f"Author {i}A", f"Author {i}B"],
                'keywords': [f"keyword_{i}_1", f"keyword_{i}_2"]
            }
            
            metadata_path = doc_dir / f"{doc_id}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
                
            docs[doc_id] = str(metadata_path)
            
        return docs
    
    def test_get_metadata(self):
        """Test retrieving metadata in different formats."""
        # Get metadata as dictionary
        metadata_dict = self.provider.get_metadata('doc_0', MetadataFormat.DICT)
        self.assertIsInstance(metadata_dict, dict)
        self.assertEqual(metadata_dict['doc_id'], 'doc_0')
        self.assertEqual(metadata_dict['title'], 'Document 0')
        
        # Get metadata as JSON
        metadata_json = self.provider.get_metadata('doc_0', MetadataFormat.JSON)
        self.assertIsInstance(metadata_json, str)
        self.assertIn('doc_0', metadata_json)
        
        # Try with non-existent document
        self.assertIsNone(self.provider.get_metadata('non_existent_doc'))
        
        # Try with invalid format
        with self.assertRaises(ValueError):
            self.provider.get_metadata('doc_0', 'invalid_format')
    
    def test_extension_points(self):
        """Test extension point management."""
        # Check registered extension points
        extension_points = self.provider.list_extension_points()
        self.assertEqual(set(extension_points), {'graphrag', 'pathrag'})
        
        # Get extension point
        graphrag_extension = self.provider.get_extension_point('graphrag')
        self.assertEqual(graphrag_extension, self.graphrag_extension)
        
        # Get non-existent extension point
        self.assertIsNone(self.provider.get_extension_point('non_existent'))
        
        # Register a new extension point
        new_extension = MockMetadataExtensionPoint('new_framework')
        self.provider.register_extension_point('new_framework', new_extension)
        
        # Check updated list
        extension_points = self.provider.list_extension_points()
        self.assertEqual(set(extension_points), {'graphrag', 'pathrag', 'new_framework'})
    
    def test_process_metadata_with_extension(self):
        """Test processing metadata with extension points."""
        # Process with GraphRAG extension
        graphrag_metadata = self.provider.process_metadata_with_extension('doc_0', 'graphrag')
        
        # Check GraphRAG-specific fields
        self.assertEqual(graphrag_metadata['framework'], 'graphrag')
        self.assertEqual(graphrag_metadata['processed_by'], 'graphrag_processor')
        self.assertEqual(graphrag_metadata['framework_specific_field'], 'value_for_graphrag')
        
        # Process with PathRAG extension
        pathrag_metadata = self.provider.process_metadata_with_extension('doc_0', 'pathrag')
        
        # Check PathRAG-specific fields
        self.assertEqual(pathrag_metadata['framework'], 'pathrag')
        self.assertEqual(pathrag_metadata['processed_by'], 'pathrag_processor')
        self.assertEqual(pathrag_metadata['framework_specific_field'], 'value_for_pathrag')
        
        # Try with non-existent extension point
        with self.assertRaises(ValueError):
            self.provider.process_metadata_with_extension('doc_0', 'non_existent')
            
        # Try with non-existent document
        with self.assertRaises(ValueError):
            self.provider.process_metadata_with_extension('non_existent_doc', 'graphrag')
    
    def test_bulk_process_metadata(self):
        """Test bulk processing of metadata."""
        # Get all document IDs
        doc_ids = list(self.sample_docs.keys())
        
        # Bulk process with GraphRAG
        results = self.provider.bulk_process_metadata(doc_ids, 'graphrag')
        
        # Check results
        self.assertEqual(len(results), 3)
        for doc_id in doc_ids:
            self.assertIn(doc_id, results)
            self.assertEqual(results[doc_id]['framework'], 'graphrag')
            
        # Try with non-existent extension point
        with self.assertRaises(ValueError):
            self.provider.bulk_process_metadata(doc_ids, 'non_existent')
            
        # Try with mix of existing and non-existing documents
        mixed_ids = doc_ids + ['non_existent_doc']
        results = self.provider.bulk_process_metadata(mixed_ids, 'graphrag')
        
        # Should only process existing documents
        self.assertEqual(len(results), 3)
        self.assertNotIn('non_existent_doc', results)


if __name__ == '__main__':
    unittest.main()
