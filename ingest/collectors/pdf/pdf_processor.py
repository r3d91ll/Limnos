"""
PDF Processor for Limnos

This module provides functionality for processing PDF documents,
extracting text content and metadata.
"""

import os
import re
import logging
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pypdf

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Processor for PDF documents."""
    
    def __init__(self):
        """Initialize the PDF processor."""
        self.logger = logging.getLogger(__name__)
    
    def process_document(self, doc_path: str) -> Dict[str, Any]:
        """
        Process a PDF document to extract text and metadata.
        
        Args:
            doc_path: Path to the PDF document
            
        Returns:
            Dictionary with document ID, text content, and metadata
        """
        doc_path = Path(doc_path)
        
        if not doc_path.suffix.lower() == '.pdf':
            self.logger.warning(f"Not a PDF file: {doc_path}")
            return self._create_error_document(doc_path, "Not a PDF file")
        
        try:
            # Extract text from PDF
            text, metadata = self._extract_text_and_metadata(doc_path)
            
            if not text:
                self.logger.warning(f"Failed to extract text from PDF: {doc_path}")
                return self._create_error_document(doc_path, "Failed to extract text")
            
            # Generate a unique document ID
            doc_id = hashlib.md5(str(doc_path).encode()).hexdigest()
            
            # Extract basic metadata
            filename = doc_path.name
            file_size = doc_path.stat().st_size
            
            # Extract title from metadata or text
            title = metadata.get('title', filename)
            if not title or title == filename:
                title = self._extract_title_from_text(text)
            
            # Extract authors from metadata or text
            authors = metadata.get('authors', [])
            if not authors:
                authors = self._extract_authors_from_text(text)
            
            # Extract abstract from text
            abstract = self._extract_abstract(text)
            
            # Create the document dictionary
            document = {
                "doc_id": doc_id,
                "text": text,
                "metadata": {
                    "filename": filename,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "path": str(doc_path),
                    "file_size": file_size,
                    "extension": ".pdf",
                    "created_at": datetime.fromtimestamp(doc_path.stat().st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(doc_path.stat().st_mtime).isoformat(),
                    "character_count": len(text),
                    "doc_type": "academic_paper"
                }
            }
            
            # Add any additional metadata from the PDF
            document["metadata"].update({k: v for k, v in metadata.items() 
                                        if k not in document["metadata"]})
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error processing document {doc_path}: {e}")
            return self._create_error_document(doc_path, str(e))
    
    def _create_error_document(self, doc_path: Path, error_message: str) -> Dict[str, Any]:
        """Create a document entry for a failed processing attempt."""
        return {
            "doc_id": hashlib.md5(str(doc_path).encode()).hexdigest(),
            "text": "",
            "metadata": {
                "filename": doc_path.name,
                "path": str(doc_path),
                "error": error_message
            }
        }
    
    def _extract_text_and_metadata(self, doc_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and metadata from a PDF file.
        
        Args:
            doc_path: Path to the PDF
            
        Returns:
            Tuple of (extracted_text, metadata_dict)
        """
        metadata = {}
        text = ""
        
        with open(doc_path, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)
            
            # Extract metadata from PDF info
            if pdf_reader.metadata:
                info = pdf_reader.metadata
                
                # Try to get title
                if hasattr(info, 'title') and info.title:
                    metadata['title'] = info.title
                
                # Try to get authors
                if hasattr(info, 'author') and info.author:
                    metadata['authors'] = [a.strip() for a in info.author.split(',')]
                
                # Try to get other standard metadata
                if hasattr(info, 'subject') and info.subject:
                    metadata['subject'] = info.subject
                
                if hasattr(info, 'keywords') and info.keywords:
                    metadata['keywords'] = [k.strip() for k in info.keywords.split(',')]
            
            # Extract text from all pages
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        
        return text, metadata
    
    def _extract_title_from_text(self, text: str) -> str:
        """
        Extract the title from document text.
        
        Args:
            text: Document text
            
        Returns:
            Extracted title or empty string
        """
        # Try common title patterns
        title_patterns = [
            r'^(?:#|Title:)\s*(.+?)$',  # Explicit title
            r'^(.+?)\n\s*(?:Abstract|Introduction)',  # First line before Abstract/Introduction
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: use the first non-empty line if it's reasonably short
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and 10 < len(line) < 100:
                return line
        
        return ""
    
    def _extract_authors_from_text(self, text: str) -> List[str]:
        """
        Extract authors from document text.
        
        Args:
            text: Document text
            
        Returns:
            List of author names
        """
        # Try common author patterns
        author_patterns = [
            r'(?:Authors?|By):\s*(.+?)$',  # Explicit authors
            r'^(.+?)\n\s*(?:\w+@\w+\.\w+)',  # Names before email addresses
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                authors_text = match.group(1).strip()
                # Split by common separators
                return [a.strip() for a in re.split(r',|\band\b|;', authors_text) if a.strip()]
        
        # Look for email addresses which might indicate authors
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text[:1000])  # Check first 1000 chars
        if emails:
            # Extract usernames from emails as potential author indicators
            return [email.split('@')[0].replace('.', ' ').title() for email in emails]
        
        return []
    
    def _extract_abstract(self, text: str) -> str:
        """
        Extract abstract from document text.
        
        Args:
            text: Document text
            
        Returns:
            Extracted abstract or empty string
        """
        # Try common abstract patterns
        abstract_patterns = [
            r'(?:Abstract|ABSTRACT)[:\s]*(.*?)(?:\n\n|\n[A-Z][a-z]+\s*\n)',  # Standard abstract section
            r'(?:Abstract|ABSTRACT)[:\s]*(.*?)(?:\n\n\d+\.|\n\n[A-Z][a-z]+\s*\n)',  # Abstract before numbered section
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: try to find the first paragraph after potential title
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1 and 100 < len(paragraphs[1]) < 1500:
            return paragraphs[1].strip()
        
        return ""
