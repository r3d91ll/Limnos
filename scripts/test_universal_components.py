#!/usr/bin/env python3
"""
Test Universal Components for Limnos.

This script provides a focused test of the universal components that form
the foundation of the Limnos framework, particularly the Universal Document Collector.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import logging

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

# Import Limnos components
try:
    from limnos.ingest.collectors.universal_collector import UniversalDocumentCollector
    from limnos.ingest.collectors.metadata_interface import MetadataExtensionPoint
    from limnos.ingest.interface import Document
    logger.info("Successfully imported Limnos components")
except ImportError as e:
    logger.error(f"Failed to import Limnos components: {e}")
    sys.exit(1)


class MockMetadataExtensionPoint(MetadataExtensionPoint):
    """Mock implementation of a metadata extension point for testing."""
    
    def __init__(self, framework_name):
        """Initialize the mock extension point."""
        self.framework_name = framework_name
        self.processed_documents = {}
        
    def transform_metadata(self, metadata):
        """Transform universal metadata into framework-specific format."""
        # Make a copy and add framework-specific fields
        framework_metadata = metadata.copy()
        framework_metadata['framework'] = self.framework_name
        framework_metadata['processed_by'] = f"{self.framework_name}_processor"
        
        # Store the processed metadata
        doc_id = metadata.get('doc_id')
        if doc_id:
            self.processed_documents[doc_id] = framework_metadata
            
        return framework_metadata


def create_sample_files(temp_dir):
    """Create sample files for testing."""
    files = {}
    
    # Create a sample PDF file (actually just a text file with .pdf extension)
    pdf_path = Path(temp_dir) / "sample.pdf"
    with open(pdf_path, 'w', encoding='utf-8') as f:
        f.write("%PDF-1.4\nThis is a sample PDF file content.\n%EOF")
    files['pdf'] = pdf_path
    
    # Create a sample text file
    txt_path = Path(temp_dir) / "sample.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("This is a sample text file content.")
    files['txt'] = txt_path
    
    # Create a sample markdown file
    md_path = Path(temp_dir) / "sample.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Sample Markdown\n\nThis is a sample markdown file content.")
    files['md'] = md_path
    
    # Create a directory with multiple files
    multi_dir = Path(temp_dir) / "multi"
    os.makedirs(multi_dir, exist_ok=True)
    
    for i in range(3):
        file_path = multi_dir / f"sample{i}.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"This is sample file {i} content.")
    
    files['multi_dir'] = multi_dir
    
    return files


def test_universal_collector():
    """Test the Universal Document Collector."""
    logger.info("Testing Universal Document Collector")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")
    
    try:
        # Create source directory
        source_dir = Path(temp_dir) / "source_documents"
        os.makedirs(source_dir, exist_ok=True)
        
        # Create sample files
        sample_files = create_sample_files(temp_dir)
        
        # Initialize the collector
        collector = UniversalDocumentCollector()
        collector.initialize({
            'source_dir': str(source_dir)
        })
        logger.info("Initialized Universal Document Collector")
        
        # Register frameworks
        collector.register_framework('graphrag')
        collector.register_framework('pathrag')
        
        # Register extension points
        graphrag_extension = MockMetadataExtensionPoint('graphrag')
        pathrag_extension = MockMetadataExtensionPoint('pathrag')
        
        collector.register_metadata_extension_point('graphrag', graphrag_extension)
        collector.register_metadata_extension_point('pathrag', pathrag_extension)
        logger.info("Registered frameworks and extension points")
        
        # Test collecting a file
        logger.info("Testing file collection")
        document, file_paths = collector.collect_file(sample_files['pdf'])
        
        # Verify document
        assert isinstance(document, Document), "Document is not an instance of Document"
        assert document.metadata.get('doc_id') is not None, "Document ID is missing"
        
        # Verify file paths
        assert 'original' in file_paths, "Original file path is missing"
        assert 'metadata' in file_paths, "Metadata file path is missing"
        assert os.path.exists(file_paths['original']), "Original file does not exist"
        assert os.path.exists(file_paths['metadata']), "Metadata file does not exist"
        
        # Get document ID
        doc_id = document.metadata['doc_id']
        
        # Test document index
        doc_index = collector.get_document_index()
        assert doc_id in doc_index, "Document not found in index"
        
        # Test metadata file
        with open(file_paths['metadata'], 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        assert metadata['doc_id'] == doc_id, "Document ID mismatch in metadata file"
        
        # Test collecting a directory
        logger.info("Testing directory collection")
        results = collector.collect_directory(sample_files['multi_dir'], recursive=True)
        
        # Verify results
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        
        # Test document list
        document_list = collector.list_documents()
        assert len(document_list) == 4, f"Expected 4 documents, got {len(document_list)}"
        
        # Test metadata extension points
        logger.info("Testing metadata extension points")
        graphrag_metadata = collector.process_metadata_with_extension(doc_id, 'graphrag')
        
        # Verify GraphRAG metadata
        assert graphrag_metadata['framework'] == 'graphrag', "Framework field missing or incorrect"
        assert graphrag_metadata['processed_by'] == 'graphrag_processor', "Processed_by field missing or incorrect"
        
        # Test deleting a document
        logger.info("Testing document deletion")
        result = collector.delete_document(doc_id)
        assert result is True, "Document deletion failed"
        
        # Verify document is gone
        document_list = collector.list_documents()
        assert doc_id not in document_list, "Document still in list after deletion"
        
        logger.info("All Universal Document Collector tests passed")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    success = test_universal_collector()
    sys.exit(0 if success else 1)
