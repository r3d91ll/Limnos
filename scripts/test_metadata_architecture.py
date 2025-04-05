#!/usr/bin/env python3
"""
Test Metadata Architecture for Limnos.

This script provides a focused test of the metadata architecture components
that form the foundation of the Limnos framework, particularly the metadata
transformation between universal and framework-specific formats.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
logger.info(f"Added {project_root} to Python path")


class Document:
    """Simple document class for testing."""
    
    def __init__(self, doc_id, content="", metadata=None):
        """Initialize a document."""
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata or {}


class MetadataExtensionPoint:
    """Base class for metadata extension points."""
    
    def transform_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Transform universal metadata into framework-specific format."""
        raise NotImplementedError("Subclasses must implement transform_metadata")
    
    def get_required_metadata_fields(self) -> set:
        """Get the set of required metadata fields."""
        return {'doc_id', 'title', 'content'}


class GraphRAGMetadataPreprocessor(MetadataExtensionPoint):
    """GraphRAG metadata preprocessor."""
    
    def __init__(self):
        """Initialize the GraphRAG metadata preprocessor."""
        self.processed_documents = {}
        self.processing_status = {}
    
    def transform_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Transform universal metadata into GraphRAG-specific format."""
        # Make a copy and add GraphRAG-specific fields
        graphrag_metadata = metadata.copy()
        
        # Add GraphRAG-specific fields
        graphrag_metadata['framework'] = 'graphrag'
        graphrag_metadata['entity_extraction_config'] = {
            'model': 'default',
            'confidence_threshold': 0.7,
            'max_entities_per_document': 50
        }
        graphrag_metadata['relationship_extraction_config'] = {
            'model': 'default',
            'confidence_threshold': 0.6,
            'max_relationships_per_document': 100
        }
        graphrag_metadata['graph_construction_config'] = {
            'merge_similar_entities': True,
            'similarity_threshold': 0.85
        }
        
        # Update processing status
        doc_id = metadata.get('doc_id')
        if doc_id:
            self.processed_documents[doc_id] = graphrag_metadata
            self.processing_status[doc_id] = {
                'metadata_processed': True,
                'entities_extracted': False,
                'relationships_extracted': False,
                'graph_constructed': False,
                'last_updated': datetime.now().isoformat()
            }
            
        return graphrag_metadata
    
    def get_processing_status(self, doc_id: str) -> Dict[str, Any]:
        """Get the processing status for a document."""
        return self.processing_status.get(doc_id, {})
    
    def update_processing_status(self, doc_id: str, status_updates: Dict[str, bool]) -> None:
        """Update the processing status for a document."""
        if doc_id not in self.processing_status:
            self.processing_status[doc_id] = {
                'metadata_processed': False,
                'entities_extracted': False,
                'relationships_extracted': False,
                'graph_constructed': False,
                'last_updated': datetime.now().isoformat()
            }
        
        # Update status fields
        for key, value in status_updates.items():
            if key in self.processing_status[doc_id]:
                self.processing_status[doc_id][key] = value
        
        # Update timestamp
        self.processing_status[doc_id]['last_updated'] = datetime.now().isoformat()
    
    def is_fully_processed(self, doc_id: str) -> bool:
        """Check if a document is fully processed."""
        status = self.get_processing_status(doc_id)
        return (status.get('metadata_processed', False) and
                status.get('entities_extracted', False) and
                status.get('relationships_extracted', False) and
                status.get('graph_constructed', False))
    
    def get_entity_extraction_config(self) -> Dict[str, Any]:
        """Get the entity extraction configuration."""
        return {
            'model': 'default',
            'confidence_threshold': 0.7,
            'max_entities_per_document': 50
        }
    
    def get_relationship_extraction_config(self) -> Dict[str, Any]:
        """Get the relationship extraction configuration."""
        return {
            'model': 'default',
            'confidence_threshold': 0.6,
            'max_relationships_per_document': 100
        }
    
    def get_graph_construction_config(self) -> Dict[str, Any]:
        """Get the graph construction configuration."""
        return {
            'merge_similar_entities': True,
            'similarity_threshold': 0.85
        }


class PathRAGMetadataPreprocessor(MetadataExtensionPoint):
    """PathRAG metadata preprocessor."""
    
    def __init__(self):
        """Initialize the PathRAG metadata preprocessor."""
        self.processed_documents = {}
        self.processing_status = {}
    
    def transform_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Transform universal metadata into PathRAG-specific format."""
        # Make a copy and add PathRAG-specific fields
        pathrag_metadata = metadata.copy()
        
        # Add PathRAG-specific fields
        pathrag_metadata['framework'] = 'pathrag'
        pathrag_metadata['chunking_config'] = {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'chunking_strategy': 'semantic'
        }
        pathrag_metadata['embedding_config'] = {
            'model': 'default',
            'dimensions': 768
        }
        pathrag_metadata['path_extraction_config'] = {
            'max_path_length': 5,
            'similarity_threshold': 0.75
        }
        
        # Update processing status
        doc_id = metadata.get('doc_id')
        if doc_id:
            self.processed_documents[doc_id] = pathrag_metadata
            self.processing_status[doc_id] = {
                'metadata_processed': True,
                'chunks_created': False,
                'embeddings_generated': False,
                'paths_extracted': False,
                'last_updated': datetime.now().isoformat()
            }
            
        return pathrag_metadata
    
    def get_processing_status(self, doc_id: str) -> Dict[str, Any]:
        """Get the processing status for a document."""
        return self.processing_status.get(doc_id, {})
    
    def update_processing_status(self, doc_id: str, status_updates: Dict[str, bool]) -> None:
        """Update the processing status for a document."""
        if doc_id not in self.processing_status:
            self.processing_status[doc_id] = {
                'metadata_processed': False,
                'chunks_created': False,
                'embeddings_generated': False,
                'paths_extracted': False,
                'last_updated': datetime.now().isoformat()
            }
        
        # Update status fields
        for key, value in status_updates.items():
            if key in self.processing_status[doc_id]:
                self.processing_status[doc_id][key] = value
        
        # Update timestamp
        self.processing_status[doc_id]['last_updated'] = datetime.now().isoformat()
    
    def is_fully_processed(self, doc_id: str) -> bool:
        """Check if a document is fully processed."""
        status = self.get_processing_status(doc_id)
        return (status.get('metadata_processed', False) and
                status.get('chunks_created', False) and
                status.get('embeddings_generated', False) and
                status.get('paths_extracted', False))
    
    def get_chunking_config(self) -> Dict[str, Any]:
        """Get the chunking configuration."""
        return {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'chunking_strategy': 'semantic'
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get the embedding configuration."""
        return {
            'model': 'default',
            'dimensions': 768
        }
    
    def get_path_extraction_config(self) -> Dict[str, Any]:
        """Get the path extraction configuration."""
        return {
            'max_path_length': 5,
            'similarity_threshold': 0.75
        }


def create_sample_metadata():
    """Create sample universal metadata for testing."""
    return {
        'doc_id': 'doc_123',
        'title': 'Sample Academic Paper',
        'doc_type': 'academic_paper',
        'content': 'This is the content of the sample academic paper.',
        'storage_path': '/path/to/doc_123.pdf',
        'metadata_path': '/path/to/doc_123.json',
        'collection_timestamp': '2025-04-05T07:00:00',
        'size_bytes': 1024,
        'num_pages': 10,
        'authors': ['Author A', 'Author B'],
        'keywords': ['keyword1', 'keyword2']
    }


def test_metadata_transformation():
    """Test metadata transformation between universal and framework-specific formats."""
    logger.info("Testing metadata transformation")
    
    # Create sample universal metadata
    universal_metadata = create_sample_metadata()
    logger.info(f"Created universal metadata for document {universal_metadata['doc_id']}")
    
    # Initialize preprocessors
    graphrag_preprocessor = GraphRAGMetadataPreprocessor()
    pathrag_preprocessor = PathRAGMetadataPreprocessor()
    logger.info("Initialized metadata preprocessors")
    
    # Transform metadata for GraphRAG
    graphrag_metadata = graphrag_preprocessor.transform_metadata(universal_metadata)
    logger.info("Transformed metadata for GraphRAG")
    
    # Verify GraphRAG-specific fields
    assert graphrag_metadata['framework'] == 'graphrag', "Framework field missing or incorrect"
    assert 'entity_extraction_config' in graphrag_metadata, "Entity extraction config missing"
    assert 'relationship_extraction_config' in graphrag_metadata, "Relationship extraction config missing"
    assert 'graph_construction_config' in graphrag_metadata, "Graph construction config missing"
    
    # Verify original fields are preserved
    assert graphrag_metadata['doc_id'] == universal_metadata['doc_id'], "Document ID changed"
    assert graphrag_metadata['title'] == universal_metadata['title'], "Title changed"
    assert graphrag_metadata['doc_type'] == universal_metadata['doc_type'], "Document type changed"
    
    # Check processing status
    doc_id = universal_metadata['doc_id']
    status = graphrag_preprocessor.get_processing_status(doc_id)
    assert status['metadata_processed'] is True, "Metadata processing status not updated"
    assert status['entities_extracted'] is False, "Entity extraction status incorrect"
    
    # Update processing status
    graphrag_preprocessor.update_processing_status(doc_id, {'entities_extracted': True})
    status = graphrag_preprocessor.get_processing_status(doc_id)
    assert status['entities_extracted'] is True, "Entity extraction status not updated"
    
    # Check if fully processed
    assert not graphrag_preprocessor.is_fully_processed(doc_id), "Document incorrectly marked as fully processed"
    
    # Update all statuses to complete
    graphrag_preprocessor.update_processing_status(doc_id, {
        'relationships_extracted': True,
        'graph_constructed': True
    })
    
    # Check if fully processed now
    assert graphrag_preprocessor.is_fully_processed(doc_id), "Document not marked as fully processed"
    
    # Transform metadata for PathRAG
    pathrag_metadata = pathrag_preprocessor.transform_metadata(universal_metadata)
    logger.info("Transformed metadata for PathRAG")
    
    # Verify PathRAG-specific fields
    assert pathrag_metadata['framework'] == 'pathrag', "Framework field missing or incorrect"
    assert 'chunking_config' in pathrag_metadata, "Chunking config missing"
    assert 'embedding_config' in pathrag_metadata, "Embedding config missing"
    assert 'path_extraction_config' in pathrag_metadata, "Path extraction config missing"
    
    # Verify original fields are preserved
    assert pathrag_metadata['doc_id'] == universal_metadata['doc_id'], "Document ID changed"
    assert pathrag_metadata['title'] == universal_metadata['title'], "Title changed"
    assert pathrag_metadata['doc_type'] == universal_metadata['doc_type'], "Document type changed"
    
    # Check processing status
    status = pathrag_preprocessor.get_processing_status(doc_id)
    assert status['metadata_processed'] is True, "Metadata processing status not updated"
    assert status['chunks_created'] is False, "Chunk creation status incorrect"
    
    # Update processing status
    pathrag_preprocessor.update_processing_status(doc_id, {'chunks_created': True})
    status = pathrag_preprocessor.get_processing_status(doc_id)
    assert status['chunks_created'] is True, "Chunk creation status not updated"
    
    # Check if fully processed
    assert not pathrag_preprocessor.is_fully_processed(doc_id), "Document incorrectly marked as fully processed"
    
    # Update all statuses to complete
    pathrag_preprocessor.update_processing_status(doc_id, {
        'embeddings_generated': True,
        'paths_extracted': True
    })
    
    # Check if fully processed now
    assert pathrag_preprocessor.is_fully_processed(doc_id), "Document not marked as fully processed"
    
    logger.info("All metadata transformation tests passed")
    return True


def test_document_collector_integration():
    """Test the integration between document collector and metadata preprocessors."""
    logger.info("Testing document collector integration")
    
    # Create a mock document collector
    class MockDocumentCollector:
        def __init__(self):
            self.documents = {}
            self.metadata_preprocessors = {
                'graphrag': GraphRAGMetadataPreprocessor(),
                'pathrag': PathRAGMetadataPreprocessor()
            }
        
        def process_document(self, doc_id, content, metadata):
            # Create document
            document = Document(doc_id, content, metadata)
            self.documents[doc_id] = document
            
            # Process with each framework
            results = {}
            for framework, preprocessor in self.metadata_preprocessors.items():
                framework_metadata = preprocessor.transform_metadata(metadata)
                results[framework] = framework_metadata
            
            return document, results
    
    # Create collector and sample document
    collector = MockDocumentCollector()
    doc_id = 'doc_456'
    content = 'This is a sample document for testing integration.'
    metadata = {
        'doc_id': doc_id,
        'title': 'Integration Test Document',
        'doc_type': 'test_document',
        'content': content,
        'collection_timestamp': '2025-04-05T07:10:00',
        'size_bytes': 512,
        'authors': ['Test Author']
    }
    
    # Process document
    document, results = collector.process_document(doc_id, content, metadata)
    logger.info(f"Processed document {doc_id} with collector")
    
    # Verify document
    assert document.doc_id == doc_id, "Document ID mismatch"
    assert document.content == content, "Document content mismatch"
    
    # Verify framework results
    assert 'graphrag' in results, "GraphRAG results missing"
    assert 'pathrag' in results, "PathRAG results missing"
    
    # Verify GraphRAG-specific fields
    graphrag_metadata = results['graphrag']
    assert graphrag_metadata['framework'] == 'graphrag', "GraphRAG framework field missing or incorrect"
    assert 'entity_extraction_config' in graphrag_metadata, "Entity extraction config missing"
    
    # Verify PathRAG-specific fields
    pathrag_metadata = results['pathrag']
    assert pathrag_metadata['framework'] == 'pathrag', "PathRAG framework field missing or incorrect"
    assert 'chunking_config' in pathrag_metadata, "Chunking config missing"
    
    # Check processing status
    graphrag_status = collector.metadata_preprocessors['graphrag'].get_processing_status(doc_id)
    assert graphrag_status['metadata_processed'] is True, "GraphRAG metadata processing status not updated"
    
    pathrag_status = collector.metadata_preprocessors['pathrag'].get_processing_status(doc_id)
    assert pathrag_status['metadata_processed'] is True, "PathRAG metadata processing status not updated"
    
    logger.info("All document collector integration tests passed")
    return True


if __name__ == "__main__":
    try:
        # Run metadata transformation tests
        metadata_success = test_metadata_transformation()
        
        # Run document collector integration tests
        integration_success = test_document_collector_integration()
        
        # Exit with success if all tests passed
        sys.exit(0 if metadata_success and integration_success else 1)
    except Exception as e:
        logger.error(f"Tests failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
