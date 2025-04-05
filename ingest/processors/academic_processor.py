"""
Academic paper processor implementation for the Limnos ingest pipeline.

This module provides a specialized implementation of the DocumentProcessor interface
that extracts rich metadata from academic papers (PDFs) including title, authors,
abstract, and sections.
"""

import os
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid
import re
from datetime import datetime

# Document format handlers
import pypdf

from limnos.ingest.interface import DocumentProcessor, Document
from limnos.ingest.processors.basic_processor import BasicDocumentProcessor
from limnos.pipeline.interfaces import Configurable
from limnos.ingest.collectors.metadata_schema import UniversalMetadataSchema, DocumentType


class AcademicPaperProcessor(BasicDocumentProcessor):
    """Academic paper processor implementation that extracts rich metadata."""
    
    def __init__(self):
        """Initialize the academic paper processor."""
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._config = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the processor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        super().initialize(config)
        
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
        # Get basic document using parent method
        document = super().process_file(file_path)
        
        # Extract enhanced academic metadata based on file type
        extension = Path(file_path).suffix.lower()
        
        enhanced_metadata = {}
        if extension == '.pdf':
            enhanced_metadata = self._extract_pdf_metadata(file_path)
        
        # Add enhanced metadata to document
        document.metadata.update(enhanced_metadata)
        
        return document

    def _extract_pdf_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract rich metadata from a PDF academic paper using the standard schema.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary of extracted metadata following the UniversalMetadataSchema
        """
        # Initialize with basic schema fields
        metadata = {
            'doc_id': str(uuid.uuid4()),
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
                
                # Extract text from first few pages for additional analysis
                text_sample = ""
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
        
        # Ensure metadata conforms to schema
        schema = UniversalMetadataSchema.from_dict(metadata)
        return schema.to_dict()
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract paper title from text."""
        # Check the first few lines for title patterns
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            for pattern in self._title_patterns:
                match = re.match(pattern, line, re.DOTALL)
                if match:
                    return match.group(1).strip()
        
        # Heuristic: First line might be the title if it's short and doesn't end with punctuation
        if lines and len(lines[0]) < 100 and not lines[0].strip().endswith(('.', ':', ';')):
            return lines[0].strip()
        
        return None
    
    def _extract_authors(self, text: str) -> Optional[List[str]]:
        """Extract paper authors from text."""
        lines = text.split('\n')
        
        # Check for author patterns
        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            for pattern in self._author_patterns:
                match = re.match(pattern, line, re.DOTALL)
                if match:
                    authors_text = match.group(1).strip()
                    # Split authors by common separators
                    authors = re.split(r',|\band\b|;', authors_text)
                    return [author.strip() for author in authors if author.strip()]
        
        # Fallback: Look for email addresses which might indicate authors
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, '\n'.join(lines[:30]))
        if emails:
            # Extract usernames from emails as potential author indicators
            return [email.split('@')[0].replace('.', ' ').title() for email in emails]
        
        return None
    
    def _extract_abstract(self, text: str) -> Optional[str]:
        """Extract paper abstract from text."""
        # Try pattern matching first
        for pattern in self._abstract_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Look for Abstract section
        abstract_match = re.search(r'(?:Abstract|ABSTRACT)[:\s]*(.*?)(?:\n\n|\n[A-Z][a-z]+\s*\n)', 
                                  text, re.DOTALL)
        if abstract_match:
            return abstract_match.group(1).strip()
        
        # Fall back to first paragraph if it's reasonably sized
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1 and 100 < len(paragraphs[1]) < 1500:
            return paragraphs[1].strip()
        
        return None
    
    def _extract_sections(self, text: str) -> List[Dict[str, str]]:
        """Extract paper sections from text."""
        sections = []
        
        # Simple section detection
        section_pattern = r'\n([A-Z][A-Za-z\s]{2,50})\n'
        section_matches = re.finditer(section_pattern, text)
        
        last_pos = 0
        for match in section_matches:
            section_title = match.group(1).strip()
            start_pos = match.end()
            
            # Skip common non-section titles
            if section_title.lower() in ['abstract', 'introduction', 'conclusion', 'references']:
                continue
                
            if last_pos > 0:
                sections.append({
                    'title': section_title,
                    'content': text[last_pos:match.start()].strip()
                })
            
            last_pos = start_pos
        
        return sections
    
    def _analyze_paper_structure(self, text: str) -> Dict[str, bool]:
        """Analyze the structure of the paper to identify key components."""
        structure = {
            'has_abstract': bool(re.search(r'\b(?:abstract|ABSTRACT)\b', text)),
            'has_introduction': bool(re.search(r'\b(?:introduction|INTRODUCTION)\b', text)),
            'has_methodology': bool(re.search(r'\b(?:methodology|method|approach|METHODOLOGY)\b', text)),
            'has_results': bool(re.search(r'\b(?:results|RESULTS|evaluation|EVALUATION)\b', text)),
            'has_discussion': bool(re.search(r'\b(?:discussion|DISCUSSION)\b', text)),
            'has_conclusion': bool(re.search(r'\b(?:conclusion|CONCLUSION)\b', text)),
            'has_references': bool(re.search(r'\b(?:references|REFERENCES|bibliography|BIBLIOGRAPHY)\b', text)),
            'has_figures': bool(re.search(r'\bfig(?:ure|\.)\s+\d+\b', text, re.IGNORECASE)),
            'has_tables': bool(re.search(r'\btable\s+\d+\b', text, re.IGNORECASE)),
            'has_equations': bool(re.search(r'\beq(?:uation|\.)\s+\d+\b', text, re.IGNORECASE)),
        }
        
        return structure
    
    def _extract_references(self, text: str) -> Optional[List[str]]:
        """Extract paper references from text."""
        # Find the references section
        ref_section_match = re.search(r'(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)[:\s]*\n(.*?)(?:\n\n\n|\Z)', 
                                     text, re.DOTALL)
        
        if not ref_section_match:
            return None
        
        ref_text = ref_section_match.group(1)
        
        # Try to separate individual references
        # This is a simple approach, for production systems consider using a citation parser
        references = []
        ref_lines = ref_text.split('\n')
        current_ref = ""
        
        for line in ref_lines:
            if re.match(r'^\[\d+\]|^\d+\.', line):  # New reference
                if current_ref:
                    references.append(current_ref.strip())
                current_ref = line
            else:
                current_ref += " " + line
        
        # Add the last reference
        if current_ref:
            references.append(current_ref.strip())
        
        return references if references else None
