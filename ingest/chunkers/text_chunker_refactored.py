"""
Text chunker implementation for the Limnos ingest pipeline (Refactored).

This module provides implementations of the DocumentChunker interface
for splitting documents into chunks based on various strategies,
using the new Pydantic models.
"""

import re
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

from limnos.ingest.processors.processor_interface import DocumentChunker
from limnos.ingest.models.document import Document, DocumentChunk
from limnos.ingest.models.metadata import UniversalMetadata, DocumentType, Section
from limnos.pipeline.interfaces import Configurable


class TextChunker(DocumentChunker):
    """Text chunker implementation that splits documents into chunks."""
    
    def __init__(self):
        """Initialize the text chunker."""
        self._logger = logging.getLogger(__name__)
        self._config = {}
        self._chunk_size = 1000
        self._chunk_overlap = 200
        self._chunk_strategy = "sliding_window"
        self._respect_sections = True
        self._respect_paragraphs = True
        self._min_chunk_size = 100
        self._initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the chunker with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config or {}
        
        # Set chunking parameters
        self._chunk_size = config.get('chunk_size', self._chunk_size)
        self._chunk_overlap = config.get('chunk_overlap', self._chunk_overlap)
        self._chunk_strategy = config.get('chunk_strategy', self._chunk_strategy)
        self._respect_sections = config.get('respect_sections', self._respect_sections)
        self._respect_paragraphs = config.get('respect_paragraphs', self._respect_paragraphs)
        self._min_chunk_size = config.get('min_chunk_size', self._min_chunk_size)
        
        # Validate configuration
        if self._chunk_overlap >= self._chunk_size:
            self._logger.warning(
                f"Chunk overlap ({self._chunk_overlap}) must be less than chunk size "
                f"({self._chunk_size}). Setting overlap to chunk_size/2."
            )
            self._chunk_overlap = self._chunk_size // 2
            
        self._initialized = True
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "document_chunker"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "document_chunker"
    
    @classmethod
    def get_plugin_name(cls) -> str:
        """Return the name of this plugin."""
        return "text"
    
    def get_default_config(self) -> Dict[str, Any]:
        """Return the default configuration for this component.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'chunk_strategy': 'sliding_window',
            'respect_sections': True,
            'respect_paragraphs': True,
            'min_chunk_size': 100
        }
    
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Chunk a document into smaller pieces.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        if not self._initialized:
            self.initialize({})
            
        # Select chunking strategy based on document characteristics
        if self._respect_sections and hasattr(document.metadata, 'sections') and document.metadata.sections:
            # If document has sections and we should respect them
            chunks = self._section_based_chunking(document)
        elif self._chunk_strategy == "sliding_window":
            chunks = self._sliding_window_chunking(document)
        elif self._chunk_strategy == "sentence":
            chunks = self._sentence_chunking(document)
        elif self._chunk_strategy == "paragraph":
            chunks = self._paragraph_chunking(document)
        else:
            self._logger.warning(
                f"Unknown chunking strategy: {self._chunk_strategy}. "
                "Falling back to sliding window."
            )
            chunks = self._sliding_window_chunking(document)
        
        return chunks
    
    def _sliding_window_chunking(self, document: Document) -> List[DocumentChunk]:
        """Chunk a document using a sliding window approach.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        content = document.content
        doc_id = document.doc_id
        
        # If content is shorter than chunk size, return a single chunk
        if len(content) <= self._chunk_size:
            return [self._create_chunk(
                document=document,
                content=content,
                chunk_index=0,
                start_char=0,
                end_char=len(content),
                strategy='sliding_window',
                total_chunks=1
            )]
        
        chunks = []
        
        # Calculate step size
        step_size = self._chunk_size - self._chunk_overlap
        
        # Generate chunks
        for i, start_idx in enumerate(range(0, len(content), step_size)):
            # If we're at the end of the document, break
            if start_idx >= len(content):
                break
            
            # Calculate end index
            end_idx = min(start_idx + self._chunk_size, len(content))
            
            # If this is the last chunk and it's too small, merge with previous
            if end_idx == len(content) and end_idx - start_idx < self._min_chunk_size:
                if chunks:  # If there are previous chunks
                    # Update the last chunk to include this content
                    last_chunk = chunks[-1]
                    last_content = last_chunk.content
                    
                    # Create a new chunk that combines the last chunk with the remaining content
                    chunks[-1] = self._create_chunk(
                        document=document,
                        content=last_content + content[start_idx:],
                        chunk_index=len(chunks) - 1,
                        start_char=last_chunk.metadata.get('start_char', 0),
                        end_char=len(content),
                        strategy='sliding_window',
                        total_chunks=0  # Will update later
                    )
                break
            
            # Create chunk
            chunk = self._create_chunk(
                document=document,
                content=content[start_idx:end_idx],
                chunk_index=i,
                start_char=start_idx,
                end_char=end_idx,
                strategy='sliding_window',
                total_chunks=0,  # Will update later
                overlap=self._chunk_overlap if i > 0 else 0
            )
            
            chunks.append(chunk)
        
        # Update total_chunks in metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def _section_based_chunking(self, document: Document) -> List[DocumentChunk]:
        """Chunk a document based on its sections.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        if not hasattr(document.metadata, 'sections') or not document.metadata.sections:
            # If no sections, fall back to sliding window
            return self._sliding_window_chunking(document)
        
        chunks = []
        
        # Process each section
        for i, section in enumerate(document.metadata.sections):
            section_content = section.content
            section_heading = section.heading
            
            # If section is short enough, keep it as one chunk
            if len(section_content) <= self._chunk_size:
                chunk = self._create_chunk(
                    document=document,
                    content=section_content,
                    chunk_index=len(chunks),
                    start_char=0,  # Relative to section
                    end_char=len(section_content),
                    strategy='section',
                    total_chunks=0,  # Will update later
                    section_heading=section_heading,
                    section_index=i
                )
                chunks.append(chunk)
            else:
                # Section is too large, apply sliding window within the section
                section_chunks = self._chunk_text(
                    document=document,
                    text=section_content,
                    chunk_index_start=len(chunks),
                    strategy='section',
                    section_heading=section_heading,
                    section_index=i
                )
                chunks.extend(section_chunks)
        
        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def _sentence_chunking(self, document: Document) -> List[DocumentChunk]:
        """Chunk a document by sentences, respecting chunk size.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        content = document.content
        
        # If content is shorter than chunk size, return a single chunk
        if len(content) <= self._chunk_size:
            return [self._create_chunk(
                document=document,
                content=content,
                chunk_index=0,
                start_char=0,
                end_char=len(content),
                strategy='sentence',
                total_chunks=1
            )]
        
        # Split content into sentences
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, content)
        
        # Remove empty sentences
        sentences = [s for s in sentences if s.strip()]
        
        return self._combine_parts_into_chunks(document, sentences, 'sentence')
    
    def _paragraph_chunking(self, document: Document) -> List[DocumentChunk]:
        """Chunk a document by paragraphs, respecting chunk size.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        content = document.content
        
        # If content is shorter than chunk size, return a single chunk
        if len(content) <= self._chunk_size:
            return [self._create_chunk(
                document=document,
                content=content,
                chunk_index=0,
                start_char=0,
                end_char=len(content),
                strategy='paragraph',
                total_chunks=1
            )]
        
        # Split content into paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', content)
        
        # Remove empty paragraphs
        paragraphs = [p for p in paragraphs if p.strip()]
        
        return self._combine_parts_into_chunks(document, paragraphs, 'paragraph')
    
    def _combine_parts_into_chunks(self, document: Document, parts: List[str], 
                                 strategy: str) -> List[DocumentChunk]:
        """Combine parts (sentences or paragraphs) into chunks of appropriate size.
        
        Args:
            document: Source document
            parts: List of parts to combine
            strategy: Chunking strategy name
            
        Returns:
            List of document chunks
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        for part in parts:
            part_length = len(part)
            
            # If this part is larger than the chunk size, split it
            if part_length > self._chunk_size:
                # If we have content in the current chunk, add it as a chunk
                if current_chunk:
                    combined_content = ' '.join(current_chunk)
                    chunk = self._create_chunk(
                        document=document,
                        content=combined_content,
                        chunk_index=len(chunks),
                        start_char=0,  # Cannot accurately track in this method
                        end_char=len(combined_content),
                        strategy=strategy,
                        total_chunks=0  # Will update later
                    )
                    chunks.append(chunk)
                    current_chunk = []
                    current_length = 0
                
                # Split the large part using sliding window
                part_chunks = self._chunk_text(
                    document=document,
                    text=part,
                    chunk_index_start=len(chunks),
                    strategy=f"{strategy}_sliding"
                )
                chunks.extend(part_chunks)
            
            # If adding this part would exceed the chunk size, start a new chunk
            elif current_length + part_length + len(current_chunk) > self._chunk_size:
                # Add current chunk if not empty
                if current_chunk:
                    combined_content = ' '.join(current_chunk)
                    chunk = self._create_chunk(
                        document=document,
                        content=combined_content,
                        chunk_index=len(chunks),
                        start_char=0,  # Cannot accurately track in this method
                        end_char=len(combined_content),
                        strategy=strategy,
                        total_chunks=0  # Will update later
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap if needed
                    if self._chunk_overlap > 0 and len(current_chunk) > 1:
                        # Add some previous parts for overlap
                        overlap_parts = []
                        overlap_length = 0
                        for prev_part in reversed(current_chunk):
                            if overlap_length + len(prev_part) <= self._chunk_overlap:
                                overlap_parts.insert(0, prev_part)
                                overlap_length += len(prev_part) + 1  # +1 for space
                            else:
                                break
                        
                        current_chunk = overlap_parts
                        current_length = overlap_length
                    else:
                        current_chunk = []
                        current_length = 0
                
                # Add current part to the new chunk
                current_chunk.append(part)
                current_length += part_length
            
            # Otherwise, add to current chunk
            else:
                current_chunk.append(part)
                current_length += part_length + 1  # +1 for space
        
        # Add the last chunk if not empty
        if current_chunk:
            combined_content = ' '.join(current_chunk)
            chunk = self._create_chunk(
                document=document,
                content=combined_content,
                chunk_index=len(chunks),
                start_char=0,  # Cannot accurately track in this method
                end_char=len(combined_content),
                strategy=strategy,
                total_chunks=0  # Will update later
            )
            chunks.append(chunk)
        
        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def _chunk_text(self, document: Document, text: str, chunk_index_start: int = 0,
                  strategy: str = 'sliding_window', **kwargs) -> List[DocumentChunk]:
        """Chunk text using sliding window approach.
        
        Args:
            document: Source document
            text: Text to chunk
            chunk_index_start: Starting index for chunks
            strategy: Chunking strategy
            **kwargs: Additional metadata to include
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        # If text is shorter than chunk size, return a single chunk
        if len(text) <= self._chunk_size:
            chunk = self._create_chunk(
                document=document,
                content=text,
                chunk_index=chunk_index_start,
                start_char=0,
                end_char=len(text),
                strategy=strategy,
                total_chunks=0,  # Will update later
                **kwargs
            )
            return [chunk]
        
        # Calculate step size
        step_size = self._chunk_size - self._chunk_overlap
        
        # Generate chunks
        chunk_index = chunk_index_start
        for start_idx in range(0, len(text), step_size):
            # If we're at the end of the text, break
            if start_idx >= len(text):
                break
            
            # Calculate end index
            end_idx = min(start_idx + self._chunk_size, len(text))
            
            # If this is the last chunk and it's too small, merge with previous
            if end_idx == len(text) and end_idx - start_idx < self._min_chunk_size:
                if chunks:  # If there are previous chunks
                    # Update the last chunk to include this content
                    last_chunk = chunks[-1]
                    last_content = last_chunk.content
                    
                    # Create a new chunk that combines the last chunk with the remaining content
                    chunks[-1] = self._create_chunk(
                        document=document,
                        content=last_content + text[start_idx:],
                        chunk_index=chunk_index - 1,
                        start_char=last_chunk.metadata.get('start_char', 0),
                        end_char=len(text),
                        strategy=strategy,
                        total_chunks=0,  # Will update later
                        **kwargs
                    )
                break
            
            # Create chunk
            chunk = self._create_chunk(
                document=document,
                content=text[start_idx:end_idx],
                chunk_index=chunk_index,
                start_char=start_idx,
                end_char=end_idx,
                strategy=strategy,
                total_chunks=0,  # Will update later
                overlap=self._chunk_overlap if chunk_index > chunk_index_start else 0,
                **kwargs
            )
            
            chunks.append(chunk)
            chunk_index += 1
        
        return chunks
    
    def _create_chunk(self, document: Document, content: str, chunk_index: int,
                    start_char: int, end_char: int, strategy: str, total_chunks: int,
                    overlap: int = 0, **kwargs) -> DocumentChunk:
        """Create a document chunk with metadata.
        
        Args:
            document: Source document
            content: Chunk content
            chunk_index: Index of the chunk
            start_char: Start character position
            end_char: End character position
            strategy: Chunking strategy
            total_chunks: Total number of chunks
            overlap: Overlap with previous chunk
            **kwargs: Additional metadata
            
        Returns:
            Document chunk
        """
        # Create basic metadata fields
        metadata = {
            'document_id': document.doc_id,
            'chunk_index': chunk_index,
            'chunk_strategy': strategy,
            'chunk_size': len(content),
            'chunk_overlap': overlap,
            'total_chunks': total_chunks,
            'start_char': start_char,
            'end_char': end_char
        }
        
        # Add document metadata fields
        if hasattr(document, 'metadata') and document.metadata:
            # Copy relevant fields from document metadata
            if hasattr(document.metadata, 'title'):
                metadata['document_title'] = document.metadata.title
                
            if hasattr(document.metadata, 'doc_type'):
                metadata['document_type'] = document.metadata.doc_type.value
                
            if hasattr(document.metadata, 'authors'):
                metadata['document_authors'] = document.metadata.authors
                
            if hasattr(document.metadata, 'source_path'):
                metadata['document_source'] = document.metadata.source_path
        
        # Add any additional metadata
        metadata.update(kwargs)
        
        # Create the chunk
        chunk_id = str(uuid.uuid4())
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=document.doc_id,
            content=content,
            metadata=metadata
        )
    
    def chunk_text(self, text: str, document_id: str, 
                  metadata: Optional[Union[Dict[str, Any], UniversalMetadata]] = None) -> List[DocumentChunk]:
        """Split text into chunks.
        
        Args:
            text: Text to chunk
            document_id: ID to associate with the chunks
            metadata: Optional metadata for the chunks
            
        Returns:
            List of document chunks
        """
        if not self._initialized:
            self.initialize({})
            
        # Create a temporary document
        if isinstance(metadata, UniversalMetadata):
            temp_document = Document(
                doc_id=document_id,
                content=text,
                metadata=metadata
            )
        else:
            # Create minimal metadata
            meta = UniversalMetadata(
                doc_id=document_id,
                title="Untitled",
                content_length=len(text)
            )
            
            # Update with provided metadata if any
            if metadata:
                for key, value in metadata.items():
                    if hasattr(meta, key):
                        setattr(meta, key, value)
            
            temp_document = Document(
                doc_id=document_id,
                content=text,
                metadata=meta
            )
        
        # Chunk the document
        return self.chunk_document(temp_document)
