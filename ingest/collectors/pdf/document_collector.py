"""
Universal Document Collector for Limnos

This module provides functionality for collecting and storing documents
with universal metadata for use across different RAG frameworks.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from limnos.ingest.collectors.pdf.pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)

class UniversalDocumentCollector:
    """
    Universal Document Collector for Limnos.
    
    Collects documents from various sources, processes them to extract
    content and metadata, and stores them in a standardized format for
    use by different RAG frameworks.
    """
    
    def __init__(self, source_dir: Optional[str] = None):
        """
        Initialize the document collector.
        
        Args:
            source_dir: Directory to store collected documents and metadata
        """
        self.logger = logging.getLogger(__name__)
        
        # Set up source documents directory
        self.source_dir = Path(source_dir) if source_dir else Path("/home/todd/ML-Lab/Olympus/limnos/data/source_documents")
        self.source_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors for different document types
        self.pdf_processor = PDFProcessor()
        
        self.logger.info(f"Initialized document collector with source dir: {self.source_dir}")
    
    def collect_file(self, file_path: Union[str, Path]) -> Tuple[Dict[str, Any], Dict[str, Path]]:
        """
        Collect a single file, process it, and store with metadata.
        
        Args:
            file_path: Path to the file to collect
            
        Returns:
            Tuple of (document_dict, file_paths) where file_paths is a dict with paths to:
            - original: Path to the stored original document
            - metadata: Path to the stored metadata JSON file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Process document based on file type
        if file_path.suffix.lower() == '.pdf':
            document = self.pdf_processor.process_document(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Create document directory using doc_id
        doc_id = document['doc_id']
        doc_dir = self.source_dir / doc_id
        doc_dir.mkdir(exist_ok=True)
        
        # Store original file
        original_filename = file_path.name
        original_path = doc_dir / original_filename
        shutil.copy2(file_path, original_path)
        
        # Update metadata with storage paths
        document['metadata']['storage_path'] = str(original_path)
        metadata_filename = f"{file_path.stem}.json"
        metadata_path = doc_dir / metadata_filename
        document['metadata']['metadata_path'] = str(metadata_path)
        
        # Store metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(document['metadata'], f, indent=2)
        
        return document, {
            'original': original_path,
            'metadata': metadata_path
        }
    
    def collect_directory(self, directory_path: Union[str, Path], 
                         recursive: bool = True,
                         file_extensions: Optional[List[str]] = None) -> List[Tuple[Dict[str, Any], Dict[str, Path]]]:
        """
        Collect all files in a directory, process them, and store with metadata.
        
        Args:
            directory_path: Path to the directory to collect
            recursive: Whether to process subdirectories
            file_extensions: Optional list of file extensions to process (default: ['.pdf'])
            
        Returns:
            List of tuples containing (document_dict, file_paths) for each collected document
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")
        
        # Default to PDF files if no extensions specified
        if not file_extensions:
            file_extensions = ['.pdf']
        
        # Normalize extensions to lowercase with leading dot
        file_extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                          for ext in file_extensions]
        
        # Find all matching files
        files = []
        if recursive:
            for ext in file_extensions:
                files.extend(directory_path.glob(f'**/*{ext}'))
        else:
            for ext in file_extensions:
                files.extend(directory_path.glob(f'*{ext}'))
        
        # Collect each file
        results = []
        for file_path in files:
            try:
                result = self.collect_file(file_path)
                results.append(result)
                self.logger.info(f"Collected {file_path}")
            except Exception as e:
                self.logger.error(f"Error collecting {file_path}: {e}")
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Path]]]:
        """
        Retrieve a previously collected document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Tuple of (document_dict, file_paths) if found, None otherwise
        """
        doc_dir = self.source_dir / doc_id
        
        if not doc_dir.exists():
            return None
        
        # Find metadata file
        metadata_files = list(doc_dir.glob('*.json'))
        if not metadata_files:
            return None
        
        metadata_path = metadata_files[0]
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Find original file
        original_files = [f for f in doc_dir.iterdir() if f.is_file() and f.suffix.lower() != '.json']
        if not original_files:
            return None
        
        original_path = original_files[0]
        
        # Create document dict
        document = {
            "doc_id": doc_id,
            "text": "",  # We don't load the text content by default to save memory
            "metadata": metadata
        }
        
        return document, {
            'original': original_path,
            'metadata': metadata_path
        }
    
    def list_documents(self) -> List[str]:
        """
        List all collected documents.
        
        Returns:
            List of document IDs
        """
        if not self.source_dir.exists():
            return []
        
        return [d.name for d in self.source_dir.iterdir() if d.is_dir()]
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a collected document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if deleted, False otherwise
        """
        doc_dir = self.source_dir / doc_id
        
        if not doc_dir.exists():
            return False
        
        # Delete directory and all contents
        shutil.rmtree(doc_dir)
        
        return True
