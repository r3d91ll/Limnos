"""
Metadata Transformation Testing Suite.

This module provides tests to verify the correct transformation of metadata
between the Universal Document Collector and each RAG framework.
"""

import os
import unittest
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List

from limnos.ingest.collectors.universal_collector import UniversalDocumentCollector
from limnos.ingest.collectors.metadata_interface import MetadataFormat
from limnos.ingest.collectors.metadata_schema import DocumentType
from limnos.implementations.graphrag.preprocessors.metadata_preprocessor import GraphRAGMetadataPreprocessor
from limnos.implementations.pathrag.preprocessors.metadata_preprocessor import PathRAGMetadataPreprocessor


class MetadataTransformationTest(unittest.TestCase):
    """Test case for metadata transformation between universal and framework-specific formats."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.source_dir = Path(self.temp_dir) / "source_documents"
        self.graphrag_dir = Path(self.temp_dir) / "implementations" / "graphrag"
        self.pathrag_dir = Path(self.temp_dir) / "implementations" / "pathrag"
        
        # Create directories
        os.makedirs(self.source_dir, exist_ok=True)
        os.makedirs(self.graphrag_dir, exist_ok=True)
        os.makedirs(self.pathrag_dir, exist_ok=True)
        
        # Initialize the Universal Document Collector
        self.collector = UniversalDocumentCollector()
        self.collector.initialize({
            'source_dir': str(self.source_dir)
        })
        
        # Initialize metadata preprocessors
        self.graphrag_preprocessor = GraphRAGMetadataPreprocessor(output_dir=self.graphrag_dir)
        self.pathrag_preprocessor = PathRAGMetadataPreprocessor(output_dir=self.pathrag_dir)
        
        # Register frameworks and preprocessors
        self.collector.register_framework('graphrag')
        self.collector.register_framework('pathrag')
        self.collector.register_metadata_extension_point('graphrag', self.graphrag_preprocessor)
        self.collector.register_metadata_extension_point('pathrag', self.pathrag_preprocessor)
        
        # Create sample documents
        self._create_sample_documents()
    
    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_documents(self):
        """Create sample documents for testing."""
        # Create academic paper document
        self._create_document(
            doc_id="doc1",
            doc_type=DocumentType.ACADEMIC_PAPER.value,
            title="Sample Academic Paper",
            content="This is a sample academic paper content.",
            authors=["Author One", "Author Two"],
            abstract="This is a sample abstract for an academic paper."
        )
        
        # Create documentation document
        self._create_document(
            doc_id="doc2",
            doc_type=DocumentType.DOCUMENTATION.value,
            title="Sample Documentation",
            content="This is sample documentation content.",
            authors=["Author Three"],
            version="1.0.0"
        )
        
        # Create code document
        self._create_document(
            doc_id="doc3",
            doc_type=DocumentType.CODE.value,
            title="Sample Code",
            content="def sample_function():\n    return 'Hello, World!'",
            authors=["Author Four"],
            language="python"
        )
    
    def _create_document(self, doc_id: str, doc_type: str, title: str, content: str, 
                        authors: List[str], **kwargs):
        """Create a sample document with the given parameters."""
        # Create document directory
        doc_dir = self.source_dir / doc_id
        os.makedirs(doc_dir, exist_ok=True)
        
        # Create metadata
        metadata = {
            'doc_id': doc_id,
            'doc_type': doc_type,
            'title': title,
            'authors': authors,
            'content': content,
            'content_length': len(content),
            'language': kwargs.get('language', 'en'),
            'storage_path': str(doc_dir / f"{doc_id}.txt"),
            'metadata_path': str(doc_dir / f"{doc_id}.json"),
            'filename': f"{doc_id}.txt",
            'extension': '.txt',
            'size_bytes': len(content),
            'keywords': kwargs.get('keywords', []),
            'categories': kwargs.get('categories', []),
            'sections': kwargs.get('sections', []),
            'custom': kwargs.get('custom', {})
        }
        
        # Add type-specific fields
        if doc_type == DocumentType.ACADEMIC_PAPER.value:
            metadata['abstract'] = kwargs.get('abstract', '')
            metadata['doi'] = kwargs.get('doi', '')
            metadata['journal'] = kwargs.get('journal', '')
            metadata['conference'] = kwargs.get('conference', '')
            metadata['references'] = kwargs.get('references', [])
        
        elif doc_type == DocumentType.DOCUMENTATION.value:
            metadata['version'] = kwargs.get('version', '')
            metadata['api_version'] = kwargs.get('api_version', '')
            metadata['framework'] = kwargs.get('framework', '')
        
        # Create content file
        with open(doc_dir / f"{doc_id}.txt", 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Create metadata file
        with open(doc_dir / f"{doc_id}.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def test_universal_metadata_access(self):
        """Test that universal metadata can be accessed correctly."""
        for doc_id in ["doc1", "doc2", "doc3"]:
            # Get metadata
            metadata = self.collector.get_metadata_provider().get_metadata(doc_id, MetadataFormat.DICT)
            
            # Check basic fields
            self.assertEqual(metadata['doc_id'], doc_id)
            self.assertIn('title', metadata)
            self.assertIn('content', metadata)
            self.assertIn('authors', metadata)
    
    def test_graphrag_metadata_transformation(self):
        """Test transformation of universal metadata to GraphRAG-specific format."""
        for doc_id in ["doc1", "doc2", "doc3"]:
            # Get universal metadata
            universal_metadata = self.collector.get_metadata_provider().get_metadata(doc_id, MetadataFormat.DICT)
            
            # Transform to GraphRAG-specific metadata
            graphrag_metadata = self.collector.process_metadata_with_extension(doc_id, 'graphrag')
            
            # Check that GraphRAG-specific fields are added
            self.assertEqual(graphrag_metadata['universal_doc_id'], doc_id)
            self.assertEqual(graphrag_metadata['framework'], 'graphrag')
            self.assertIn('entity_extraction_status', graphrag_metadata)
            self.assertIn('relationship_extraction_status', graphrag_metadata)
            self.assertIn('graph_construction_status', graphrag_metadata)
            self.assertIn('extraction_hints', graphrag_metadata)
            self.assertIn('graph_params', graphrag_metadata)
            
            # Check that universal fields are preserved
            self.assertEqual(graphrag_metadata['doc_id'], universal_metadata['doc_id'])
            self.assertEqual(graphrag_metadata['title'], universal_metadata['title'])
            self.assertEqual(graphrag_metadata['content'], universal_metadata['content'])
            
            # Check document type-specific extraction hints
            doc_type = universal_metadata['doc_type']
            if doc_type == DocumentType.ACADEMIC_PAPER.value:
                self.assertIn('method', graphrag_metadata['extraction_hints']['entity_types'])
                self.assertIn('evaluates', graphrag_metadata['extraction_hints']['relationship_types'])
                self.assertIn('abstract', graphrag_metadata['extraction_hints']['prioritize_sections'])
            
            elif doc_type == DocumentType.DOCUMENTATION.value:
                self.assertIn('function', graphrag_metadata['extraction_hints']['entity_types'])
                self.assertIn('calls', graphrag_metadata['extraction_hints']['relationship_types'])
                self.assertIn('api', graphrag_metadata['extraction_hints']['prioritize_sections'])
            
            elif doc_type == DocumentType.CODE.value:
                self.assertIn('class', graphrag_metadata['extraction_hints']['entity_types'])
                self.assertIn('imports', graphrag_metadata['extraction_hints']['relationship_types'])
    
    def test_pathrag_metadata_transformation(self):
        """Test transformation of universal metadata to PathRAG-specific format."""
        for doc_id in ["doc1", "doc2", "doc3"]:
            # Get universal metadata
            universal_metadata = self.collector.get_metadata_provider().get_metadata(doc_id, MetadataFormat.DICT)
            
            # Transform to PathRAG-specific metadata
            pathrag_metadata = self.collector.process_metadata_with_extension(doc_id, 'pathrag')
            
            # Check that PathRAG-specific fields are added
            self.assertEqual(pathrag_metadata['universal_doc_id'], doc_id)
            self.assertEqual(pathrag_metadata['framework'], 'pathrag')
            self.assertIn('chunking_status', pathrag_metadata)
            self.assertIn('embedding_status', pathrag_metadata)
            self.assertIn('path_extraction_status', pathrag_metadata)
            self.assertIn('chunking_params', pathrag_metadata)
            self.assertIn('path_params', pathrag_metadata)
            
            # Check that universal fields are preserved
            self.assertEqual(pathrag_metadata['doc_id'], universal_metadata['doc_id'])
            self.assertEqual(pathrag_metadata['title'], universal_metadata['title'])
            self.assertEqual(pathrag_metadata['content'], universal_metadata['content'])
            
            # Check document type-specific chunking parameters
            doc_type = universal_metadata['doc_type']
            if doc_type == DocumentType.ACADEMIC_PAPER.value:
                self.assertEqual(pathrag_metadata['chunking_params']['chunk_size'], 800)
                self.assertEqual(pathrag_metadata['chunking_params']['chunk_overlap'], 150)
                self.assertTrue(pathrag_metadata['chunking_params']['respect_sections'])
                self.assertIn('sequential', pathrag_metadata['path_params']['path_types'])
                self.assertIn('citation', pathrag_metadata['path_params']['path_types'])
            
            elif doc_type == DocumentType.DOCUMENTATION.value:
                self.assertEqual(pathrag_metadata['chunking_params']['chunk_size'], 600)
                self.assertEqual(pathrag_metadata['chunking_params']['chunk_overlap'], 100)
                self.assertTrue(pathrag_metadata['chunking_params']['respect_sections'])
                self.assertIn('reference', pathrag_metadata['path_params']['path_types'])
            
            elif doc_type == DocumentType.CODE.value:
                self.assertEqual(pathrag_metadata['chunking_params']['chunk_size'], 400)
                self.assertEqual(pathrag_metadata['chunking_params']['chunk_overlap'], 50)
                self.assertFalse(pathrag_metadata['chunking_params']['respect_sections'])
                self.assertEqual(pathrag_metadata['chunking_params']['chunk_strategy'], 'by_function')
                self.assertIn('call_graph', pathrag_metadata['path_params']['path_types'])
    
    def test_metadata_storage_separation(self):
        """Test that framework-specific metadata is stored in separate directories."""
        for doc_id in ["doc1", "doc2", "doc3"]:
            # Process metadata for both frameworks
            self.collector.process_metadata_with_extension(doc_id, 'graphrag')
            self.collector.process_metadata_with_extension(doc_id, 'pathrag')
            
            # Check that metadata is stored in the correct directories
            graphrag_metadata_path = self.graphrag_dir / "metadata" / f"{doc_id}.json"
            pathrag_metadata_path = self.pathrag_dir / "metadata" / f"{doc_id}.json"
            
            self.assertTrue(graphrag_metadata_path.exists())
            self.assertTrue(pathrag_metadata_path.exists())
            
            # Check that the metadata contains the correct framework identifier
            with open(graphrag_metadata_path, 'r', encoding='utf-8') as f:
                graphrag_metadata = json.load(f)
                self.assertEqual(graphrag_metadata['framework'], 'graphrag')
            
            with open(pathrag_metadata_path, 'r', encoding='utf-8') as f:
                pathrag_metadata = json.load(f)
                self.assertEqual(pathrag_metadata['framework'], 'pathrag')
    
    def test_custom_metadata_fields(self):
        """Test that custom metadata fields are preserved during transformation."""
        # Create a document with custom metadata fields
        self._create_document(
            doc_id="doc4",
            doc_type=DocumentType.ACADEMIC_PAPER.value,
            title="Custom Metadata Paper",
            content="This paper has custom metadata fields.",
            authors=["Custom Author"],
            custom={
                'entity_types': ['custom_entity_type'],
                'relationship_types': ['custom_relationship_type'],
                'chunking_params': {
                    'chunk_size': 1500,
                    'chunk_overlap': 300
                },
                'path_params': {
                    'max_path_length': 10
                }
            }
        )
        
        # Process metadata for both frameworks
        graphrag_metadata = self.collector.process_metadata_with_extension('doc4', 'graphrag')
        pathrag_metadata = self.collector.process_metadata_with_extension('doc4', 'pathrag')
        
        # Check that custom entity types are included in GraphRAG metadata
        self.assertIn('custom_entity_type', graphrag_metadata['extraction_hints']['entity_types'])
        self.assertIn('custom_relationship_type', graphrag_metadata['extraction_hints']['relationship_types'])
        
        # Check that custom chunking parameters are included in PathRAG metadata
        self.assertEqual(pathrag_metadata['chunking_params']['chunk_size'], 1500)
        self.assertEqual(pathrag_metadata['chunking_params']['chunk_overlap'], 300)
        self.assertEqual(pathrag_metadata['path_params']['max_path_length'], 10)
    
    def test_metadata_extension_points(self):
        """Test that metadata extension points are registered and accessible."""
        # Check that extension points are registered
        extension_points = self.collector.get_metadata_provider().list_extension_points()
        self.assertIn('graphrag', extension_points)
        self.assertIn('pathrag', extension_points)
        
        # Check that extension points can be retrieved
        graphrag_extension = self.collector.get_metadata_provider().get_extension_point('graphrag')
        pathrag_extension = self.collector.get_metadata_provider().get_extension_point('pathrag')
        
        self.assertIsNotNone(graphrag_extension)
        self.assertIsNotNone(pathrag_extension)
        self.assertEqual(graphrag_extension, self.graphrag_preprocessor)
        self.assertEqual(pathrag_extension, self.pathrag_preprocessor)
    
    def test_bulk_metadata_processing(self):
        """Test bulk processing of metadata for multiple documents."""
        doc_ids = ["doc1", "doc2", "doc3"]
        
        # Bulk process for GraphRAG
        graphrag_results = self.collector.bulk_process_metadata(doc_ids, 'graphrag')
        
        # Check that all documents were processed
        self.assertEqual(len(graphrag_results), 3)
        for doc_id in doc_ids:
            self.assertIn(doc_id, graphrag_results)
            self.assertEqual(graphrag_results[doc_id]['framework'], 'graphrag')
        
        # Bulk process for PathRAG
        pathrag_results = self.collector.bulk_process_metadata(doc_ids, 'pathrag')
        
        # Check that all documents were processed
        self.assertEqual(len(pathrag_results), 3)
        for doc_id in doc_ids:
            self.assertIn(doc_id, pathrag_results)
            self.assertEqual(pathrag_results[doc_id]['framework'], 'pathrag')


if __name__ == '__main__':
    unittest.main()
