"""
Document Collector Integration Test.

This module tests the integration between the Universal Document Collector,
metadata preprocessors, and the GraphRAG/PathRAG frameworks.
"""

import os
import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from limnos.ingest.collectors.universal_collector import UniversalDocumentCollector
from limnos.ingest.collectors.metadata_interface import MetadataFormat
from limnos.ingest.collectors.metadata_schema import DocumentType
from limnos.implementations.graphrag.preprocessors.metadata_preprocessor import GraphRAGMetadataPreprocessor
from limnos.implementations.pathrag.preprocessors.metadata_preprocessor import PathRAGMetadataPreprocessor
from limnos.implementations.graphrag.integrations.document_collector.document_collector_integration import DocumentCollectorIntegration as GraphRAGDocumentCollectorIntegration


class DocumentCollectorIntegrationTest(unittest.TestCase):
    """Test case for integration between document collector and RAG frameworks."""
    
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
        
        # Mock components for GraphRAG integration
        self.entity_extractor_mock = MagicMock()
        self.relationship_extractor_mock = MagicMock()
        self.graph_constructor_mock = MagicMock()
        
        # Initialize GraphRAG integration
        self.graphrag_integration = GraphRAGDocumentCollectorIntegration(
            universal_collector=self.collector,
            entity_extractor=self.entity_extractor_mock,
            relationship_extractor=self.relationship_extractor_mock,
            graph_constructor=self.graph_constructor_mock,
            output_dir=self.graphrag_dir,
            config={
                'force_reprocess': True  # Force reprocessing for testing
            }
        )
    
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
    
    def _create_document(self, doc_id: str, doc_type: str, title: str, content: str, 
                        authors: list, **kwargs):
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
    
    def test_metadata_preprocessing(self):
        """Test that metadata preprocessing works correctly."""
        # Process metadata for both frameworks
        graphrag_metadata = self.collector.process_metadata_with_extension("doc1", 'graphrag')
        pathrag_metadata = self.collector.process_metadata_with_extension("doc1", 'pathrag')
        
        # Check that framework-specific metadata was created
        self.assertEqual(graphrag_metadata['framework'], 'graphrag')
        self.assertEqual(pathrag_metadata['framework'], 'pathrag')
        
        # Check that framework-specific fields were added
        self.assertIn('entity_extraction_status', graphrag_metadata)
        self.assertIn('relationship_extraction_status', graphrag_metadata)
        self.assertIn('graph_construction_status', graphrag_metadata)
        
        self.assertIn('chunking_status', pathrag_metadata)
        self.assertIn('embedding_status', pathrag_metadata)
        self.assertIn('path_extraction_status', pathrag_metadata)
    
    def test_document_processing(self):
        """Test that document processing works correctly."""
        # Setup mock returns
        self.entity_extractor_mock.extract_entities.return_value = [
            {'id': 'entity1', 'type': 'concept', 'name': 'Sample Concept'},
            {'id': 'entity2', 'type': 'term', 'name': 'Sample Term'}
        ]
        self.relationship_extractor_mock.extract_relationships.return_value = [
            {'id': 'rel1', 'source': 'entity1', 'target': 'entity2', 'type': 'mentions'}
        ]
        graph_mock = MagicMock()
        graph_mock.number_of_nodes.return_value = 2
        graph_mock.number_of_edges.return_value = 1
        self.graph_constructor_mock.build_graph.return_value = graph_mock
        
        # Process document
        result = self.graphrag_integration.process_document("doc1")
        
        # Check result
        self.assertTrue(result['success'])
        self.assertEqual(result['entity_count'], 2)
        self.assertEqual(result['relationship_count'], 1)
        self.assertEqual(result['graph_node_count'], 2)
        self.assertEqual(result['graph_edge_count'], 1)
        
        # Check that metadata preprocessor was used
        self.entity_extractor_mock.extract_entities.assert_called_once()
        self.relationship_extractor_mock.extract_relationships.assert_called_once()
        self.graph_constructor_mock.build_graph.assert_called_once()
    
    def test_processing_status_tracking(self):
        """Test that processing status is tracked correctly."""
        # Setup mock returns
        self.entity_extractor_mock.extract_entities.return_value = [
            {'id': 'entity1', 'type': 'concept', 'name': 'Sample Concept'}
        ]
        self.relationship_extractor_mock.extract_relationships.return_value = [
            {'id': 'rel1', 'source': 'entity1', 'target': 'doc1', 'type': 'mentions'}
        ]
        graph_mock = MagicMock()
        graph_mock.number_of_nodes.return_value = 2
        graph_mock.number_of_edges.return_value = 1
        self.graph_constructor_mock.build_graph.return_value = graph_mock
        
        # Process document
        self.graphrag_integration.process_document("doc1")
        
        # Check processing status
        status = self.graphrag_preprocessor.get_processing_status("doc1")
        self.assertEqual(status['entity_extraction'], 'completed')
        self.assertEqual(status['relationship_extraction'], 'completed')
        self.assertEqual(status['graph_construction'], 'completed')
        
        # Check is_document_processed
        self.assertTrue(self.graphrag_preprocessor.is_document_processed("doc1"))
    
    def test_error_handling(self):
        """Test that errors are handled correctly during document processing."""
        # Setup mock to raise an exception
        self.entity_extractor_mock.extract_entities.side_effect = Exception("Test error")
        
        # Process document
        result = self.graphrag_integration.process_document("doc1")
        
        # Check result
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['error'])
        self.assertIn("Test error", result['error'])
        
        # Check processing status
        status = self.graphrag_preprocessor.get_processing_status("doc1")
        self.assertEqual(status['entity_extraction'], 'failed')
        self.assertEqual(status['relationship_extraction'], 'failed')
        self.assertEqual(status['graph_construction'], 'failed')
        
        # Check is_document_processed
        self.assertFalse(self.graphrag_preprocessor.is_document_processed("doc1"))
    
    def test_bulk_document_processing(self):
        """Test that bulk document processing works correctly."""
        # Setup mock returns
        self.entity_extractor_mock.extract_entities.return_value = [
            {'id': 'entity1', 'type': 'concept', 'name': 'Sample Concept'}
        ]
        self.relationship_extractor_mock.extract_relationships.return_value = [
            {'id': 'rel1', 'source': 'entity1', 'target': 'doc1', 'type': 'mentions'}
        ]
        graph_mock = MagicMock()
        graph_mock.number_of_nodes.return_value = 2
        graph_mock.number_of_edges.return_value = 1
        self.graph_constructor_mock.build_graph.return_value = graph_mock
        
        # Process documents in bulk
        results = self.graphrag_integration.process_documents(["doc1", "doc2"])
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertTrue(results["doc1"]['success'])
        self.assertTrue(results["doc2"]['success'])
        
        # Check processing status for both documents
        self.assertTrue(self.graphrag_preprocessor.is_document_processed("doc1"))
        self.assertTrue(self.graphrag_preprocessor.is_document_processed("doc2"))


if __name__ == '__main__':
    unittest.main()
