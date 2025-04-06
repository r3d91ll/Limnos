"""
Academic paper processor implementation for the Limnos ingest pipeline (Refactored).

This module provides a specialized implementation of the DocumentProcessor interface
that extracts rich metadata from academic papers (PDFs) including title, authors,
abstract, and sections. This refactored version uses Pydantic models.
"""

import os
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import re
from datetime import datetime

# Document format handlers
import pypdf

from limnos.ingest.processors.processor_interface import DocumentProcessor
from limnos.ingest.models.document import Document
from limnos.ingest.models.metadata import UniversalMetadata, DocumentType, Section
from limnos.pipeline.interfaces import Configurable


class AcademicPaperProcessor(DocumentProcessor):
    """Academic paper processor implementation that extracts rich metadata."""
    
    def __init__(self):
        """Initialize the academic paper processor."""
        self._logger = logging.getLogger(__name__)
        self._config = {}
        self._initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the processor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config or {}
        
        # Academic processor specific config
        self._title_patterns = config.get('title_patterns', [
            r'^#\s+(.+)$',              # Markdown heading
            r'^\s*Title:\s*(.+)$',      # Explicit Title field
            r'^.*?\\title\{(.+?)\}',    # LaTeX title
        ])
        
        self._author_patterns = config.get('author_patterns', [
            r'^\s*Authors?:\s*(.+)$',   # Explicit Authors field 
            r'^.*?\\author\{(.+?)\}',   # LaTeX author
        ])
        
        self._abstract_patterns = config.get('abstract_patterns', [
            r'^\s*Abstract:?\s*(.+?)\n\n',  # Explicit Abstract section
            r'^\s*ABSTRACT\s*(.+?)\n\n',    # Uppercase ABSTRACT
            r'\\begin\{abstract\}(.+?)\\end\{abstract\}',  # LaTeX abstract
        ])
        
        self._initialized = True
    
    @classmethod
    def get_plugin_name(cls) -> str:
        """Return the name of this plugin."""
        return "academic"
        
    def get_default_config(self) -> Dict[str, Any]:
        """Return the default configuration for this component.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'title_patterns': [
                r'^#\s+(.+)$',              # Markdown heading
                r'^\s*Title:\s*(.+)$',      # Explicit Title field
                r'^.*?\\title\{(.+?)\}',    # LaTeX title
            ],
            'author_patterns': [
                r'^\s*Authors?:\s*(.+)$',   # Explicit Authors field 
                r'^.*?\\author\{(.+?)\}',   # LaTeX author
            ],
            'abstract_patterns': [
                r'^\s*Abstract:?\s*(.+?)\n\n',  # Explicit Abstract section
                r'^\s*ABSTRACT\s*(.+?)\n\n',    # Uppercase ABSTRACT
                r'\\begin\{abstract\}(.+?)\\end\{abstract\}',  # LaTeX abstract
            ]
        }
    
    def process_file(self, file_path: Path) -> Document:
        """Process an academic paper file with enhanced metadata extraction.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Processed Document with rich academic metadata
        """
        if not self._initialized:
            self.initialize({})
        
        try:
            # Start with basic metadata
            doc_id = str(uuid.uuid4())
            content = self._extract_text_content(file_path)
            
            # Extract enhanced academic metadata
            metadata_dict = self._extract_metadata(file_path, content)
            
            # Create Pydantic UniversalMetadata model
            metadata = UniversalMetadata(
                doc_id=doc_id,
                doc_type=DocumentType.ACADEMIC_PAPER,
                title=metadata_dict.get('title', file_path.stem),
                file_path=str(file_path),
                storage_path=None,  # Will be set by the collector
                content_length=len(content),
                filename=file_path.name,
                extension=file_path.suffix.lower(),
                size_bytes=file_path.stat().st_size,
                last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                source_path=str(file_path),
                language=metadata_dict.get('language', 'en'),
                authors=metadata_dict.get('authors', []),
                abstract=metadata_dict.get('abstract', ''),
                keywords=metadata_dict.get('keywords', []),
                sections=[
                    Section(
                        heading=section.get('heading', ''),
                        content=section.get('content', ''),
                        level=section.get('level', 1)
                    ) for section in metadata_dict.get('sections', [])
                ]
            )
            
            # Create document with extracted content and metadata
            document = Document(
                doc_id=doc_id,
                content=content,
                metadata=metadata,
                file_path=file_path
            )
            
            return document
            
        except Exception as e:
            self._logger.error(f"Error processing file {file_path}: {e}")
            
            # Create a minimal document if extraction fails
            doc_id = str(uuid.uuid4())
            
            # Create basic metadata
            metadata = UniversalMetadata(
                doc_id=doc_id,
                doc_type=DocumentType.ACADEMIC_PAPER,
                title=file_path.stem,
                file_path=str(file_path),
                storage_path=None,
                content_length=0,
                filename=file_path.name,
                extension=file_path.suffix.lower(),
                size_bytes=file_path.stat().st_size,
                last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                source_path=str(file_path),
                language='en'
            )
            
            # Create document with minimal content
            document = Document(
                doc_id=doc_id,
                content="",
                metadata=metadata,
                file_path=file_path
            )
            
            return document
    
    def _extract_text_content(self, file_path: Path) -> str:
        """Extract text content from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
        """
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self._extract_pdf_text(file_path)
        elif extension in ['.txt', '.md', '.markdown']:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        else:
            self._logger.warning(f"Unsupported file extension for academic processor: {extension}")
            return ""
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                
                # Extract text from all pages
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n\n"
                
                return text
                
        except Exception as e:
            self._logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    def _extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract metadata from a file.
        
        Args:
            file_path: Path to the file
            content: Extracted text content
            
        Returns:
            Dictionary of extracted metadata
        """
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self._extract_pdf_metadata(file_path, content)
        else:
            return self._extract_text_metadata(content)
    
    def _extract_pdf_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract rich metadata from a PDF academic paper.
        
        Args:
            file_path: Path to the PDF file
            content: Extracted text content
            
        Returns:
            Dictionary of extracted metadata
        """
        # Initialize with basic fields
        metadata = {
            'doc_type': DocumentType.ACADEMIC_PAPER.value,
            'filename': file_path.name,
            'extension': file_path.suffix.lower(),
            'size_bytes': file_path.stat().st_size,
            'last_modified': file_path.stat().st_mtime,
            'date_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'source_path': str(file_path),
            'language': 'en'  # Default assumption
        }
        
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                
                # Get metadata from PDF info
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
                
                # Extract text from first few pages for additional analysis if content not provided
                text_sample = content if content else ""
                if not text_sample:
                    max_pages = min(5, len(pdf_reader.pages))
                    for page_num in range(max_pages):
                        page = pdf_reader.pages[page_num]
                        text_sample += page.extract_text() + "\n\n"
                
                # Extract title if not found in metadata
                if 'title' not in metadata:
                    title = self._extract_title(text_sample)
                    if title:
                        metadata['title'] = title
                
                # Extract authors if not found in metadata
                if 'authors' not in metadata:
                    authors = self._extract_authors(text_sample)
                    if authors:
                        metadata['authors'] = authors
                
                # Extract abstract
                abstract = self._extract_abstract(text_sample)
                if abstract:
                    metadata['abstract'] = abstract
                
                # Extract sections
                sections = self._extract_sections(text_sample)
                if sections:
                    metadata['sections'] = sections
                
                # Add paper structure
                metadata['structure'] = self._analyze_paper_structure(text_sample)
                
                # Extract references (if available)
                references = self._extract_references(text_sample)
                if references:
                    metadata['references'] = references
                
        except Exception as e:
            self._logger.error(f"Error extracting metadata from PDF {file_path}: {e}")
        
        return metadata
    
    def _extract_text_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from text content.
        
        Args:
            content: Text content
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {
            'doc_type': DocumentType.ACADEMIC_PAPER.value,
            'language': 'en'  # Default assumption
        }
        
        # Extract title
        title = self._extract_title(content)
        if title:
            metadata['title'] = title
        
        # Extract authors
        authors = self._extract_authors(content)
        if authors:
            metadata['authors'] = authors
        
        # Extract abstract
        abstract = self._extract_abstract(content)
        if abstract:
            metadata['abstract'] = abstract
        
        # Extract sections
        sections = self._extract_sections(content)
        if sections:
            metadata['sections'] = sections
        
        # Extract references
        references = self._extract_references(content)
        if references:
            metadata['references'] = references
        
        return metadata
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract the title from text.
        
        Args:
            text: Text to extract title from
            
        Returns:
            Extracted title if found, None otherwise
        """
        # Try to extract title using patterns
        for pattern in self._title_patterns:
            matches = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if matches:
                return matches.group(1).strip()
        
        # If not found with patterns, try to extract the first heading
        lines = text.split('\n')
        for line in lines[:20]:  # Look in first 20 lines
            if line.strip() and not line.startswith('#'):
                return line.strip()
        
        return None
    
    def _extract_authors(self, text: str) -> List[str]:
        """Extract authors from text.
        
        Args:
            text: Text to extract authors from
            
        Returns:
            List of extracted authors
        """
        # Try to extract authors using patterns
        for pattern in self._author_patterns:
            matches = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if matches:
                authors_str = matches.group(1).strip()
                # Split by common separators
                authors = re.split(r',|\band\b|;', authors_str)
                return [author.strip() for author in authors if author.strip()]
        
        return []
    
    def _extract_abstract(self, text: str) -> Optional[str]:
        """Extract abstract from text.
        
        Args:
            text: Text to extract abstract from
            
        Returns:
            Extracted abstract if found, None otherwise
        """
        # Try to extract abstract using patterns
        for pattern in self._abstract_patterns:
            matches = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if matches:
                return matches.group(1).strip()
        
        # Try to find abstract section
        abstract_section = None
        sections = self._extract_sections(text)
        for section in sections:
            if section.get('heading', '').lower() == 'abstract':
                abstract_section = section.get('content', '')
                break
        
        return abstract_section
    
    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract sections from text.
        
        Args:
            text: Text to extract sections from
            
        Returns:
            List of sections with heading, content, and level
        """
        sections = []
        
        # Match Markdown headings
        section_pattern = r'^(#{1,6})\s+(.+?)\s*$'
        matches = re.finditer(section_pattern, text, re.MULTILINE)
        
        last_end = 0
        for i, match in enumerate(matches):
            heading = match.group(2).strip()
            level = len(match.group(1))
            start = match.end()
            
            # If this is not the first section, set the end of the previous section
            if i > 0:
                sections[-1]['content'] = text[sections[-1]['start']:match.start()].strip()
            
            # Add the new section
            sections.append({
                'heading': heading,
                'level': level,
                'start': start
            })
            
            last_end = start
        
        # Set the content of the last section
        if sections:
            sections[-1]['content'] = text[sections[-1]['start']:].strip()
        
        # Remove temporary 'start' field and only return the relevant fields
        return [{k: v for k, v in section.items() if k != 'start'} for section in sections]
    
    def _analyze_paper_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the structure of an academic paper.
        
        Args:
            text: Paper text content
            
        Returns:
            Dictionary with structure information
        """
        structure = {
            'has_abstract': bool(self._extract_abstract(text)),
            'has_introduction': False,
            'has_methodology': False,
            'has_results': False,
            'has_discussion': False,
            'has_conclusion': False,
            'has_references': bool(self._extract_references(text)),
            'section_count': 0
        }
        
        # Check for standard sections
        sections = self._extract_sections(text)
        structure['section_count'] = len(sections)
        
        for section in sections:
            heading = section.get('heading', '').lower()
            
            if 'introduction' in heading:
                structure['has_introduction'] = True
            elif any(term in heading for term in ['method', 'methodology', 'approach']):
                structure['has_methodology'] = True
            elif 'results' in heading:
                structure['has_results'] = True
            elif 'discussion' in heading:
                structure['has_discussion'] = True
            elif 'conclusion' in heading:
                structure['has_conclusion'] = True
        
        return structure
    
    def _extract_references(self, text: str) -> List[Dict[str, Any]]:
        """Extract references from text.
        
        Args:
            text: Text to extract references from
            
        Returns:
            List of references
        """
        references = []
        
        # Look for a references or bibliography section
        ref_section = None
        ref_pattern = r'(?:^|\n)(?:#{1,6}\s+)?(?:References|Bibliography).*?\n([\s\S]+?)(?:\n#{1,6}|\Z)'
        ref_match = re.search(ref_pattern, text, re.IGNORECASE)
        
        if ref_match:
            ref_section = ref_match.group(1).strip()
            
            # Try to split references by common patterns
            ref_entries = re.split(r'\n\s*\[\d+\]|\n\s*\d+\.|\n\n', ref_section)
            
            for entry in ref_entries:
                entry = entry.strip()
                if entry:
                    # Try to extract author and title
                    author_match = re.search(r'^([^\.]+)\.', entry)
                    title_match = re.search(r'\.([^\.]+)\.', entry)
                    
                    reference = {'raw': entry}
                    
                    if author_match:
                        reference['authors'] = author_match.group(1).strip()
                    
                    if title_match:
                        reference['title'] = title_match.group(1).strip()
                    
                    references.append(reference)
        
        return references
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Process text and convert it to a document.
        
        Args:
            text: Text to process
            metadata: Optional metadata for the document
            
        Returns:
            Processed document
        """
        if not self._initialized:
            self.initialize({})
        
        # Generate a document ID
        doc_id = metadata.get('doc_id', str(uuid.uuid4())) if metadata else str(uuid.uuid4())
        
        # Extract metadata from text
        extracted_metadata = self._extract_text_metadata(text)
        
        # Merge with provided metadata if any
        if metadata:
            extracted_metadata.update(metadata)
        
        # Ensure required fields are present
        if 'title' not in extracted_metadata:
            extracted_metadata['title'] = 'Untitled Document'
        
        if 'doc_type' not in extracted_metadata:
            extracted_metadata['doc_type'] = DocumentType.ACADEMIC_PAPER.value
        
        # Create Pydantic UniversalMetadata model
        universal_metadata = UniversalMetadata(
            doc_id=doc_id,
            doc_type=DocumentType.ACADEMIC_PAPER,
            title=extracted_metadata.get('title', 'Untitled Document'),
            content_length=len(text),
            language=extracted_metadata.get('language', 'en'),
            authors=extracted_metadata.get('authors', []),
            abstract=extracted_metadata.get('abstract', ''),
            keywords=extracted_metadata.get('keywords', []),
            sections=[
                Section(
                    heading=section.get('heading', ''),
                    content=section.get('content', ''),
                    level=section.get('level', 1)
                ) for section in extracted_metadata.get('sections', [])
            ]
        )
        
        # Create document with text and metadata
        document = Document(
            doc_id=doc_id,
            content=text,
            metadata=universal_metadata
        )
        
        return document
    
    def process_directory(self, directory_path: Path, 
                         recursive: bool = True, 
                         file_extensions: Optional[List[str]] = None) -> List[Document]:
        """Process all files in a directory and convert them to documents.
        
        Args:
            directory_path: Path to the directory to process
            recursive: Whether to process subdirectories
            file_extensions: Optional list of file extensions to process
            
        Returns:
            List of processed documents
        """
        if not self._initialized:
            self.initialize({})
        
        documents = []
        
        # Default to PDF files if no extensions specified
        if not file_extensions:
            file_extensions = ['.pdf', '.txt', '.md', '.markdown']
        
        # Normalize extensions
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in file_extensions]
        
        # Get all matching files
        if recursive:
            for ext in extensions:
                for file_path in directory_path.glob(f'**/*{ext}'):
                    try:
                        document = self.process_file(file_path)
                        documents.append(document)
                    except Exception as e:
                        self._logger.error(f"Error processing file {file_path}: {e}")
        else:
            for ext in extensions:
                for file_path in directory_path.glob(f'*{ext}'):
                    try:
                        document = self.process_file(file_path)
                        documents.append(document)
                    except Exception as e:
                        self._logger.error(f"Error processing file {file_path}: {e}")
        
        return documents
