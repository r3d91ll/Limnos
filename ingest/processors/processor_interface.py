"""
Document processor interfaces for the Limnos framework.

This module defines the refactored interfaces for document processors
using the new Pydantic models for Document and Metadata.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from pathlib import Path

from limnos.ingest.models.document import Document, DocumentChunk
from limnos.ingest.models.metadata import UniversalMetadata, DocumentType
from limnos.pipeline.interfaces import Component, Configurable, Pluggable


class DocumentProcessor(Component):
    """Interface for document processors in the ingest pipeline.
    
    Document processors are responsible for extracting content and metadata from files
    and converting them into the Document model with appropriate metadata.
    """
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "document_processor"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "document_processor"
    
    @abstractmethod
    def process_file(self, file_path: Path) -> Document:
        """Process a file and convert it to a document.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Processed document
        """
        pass
    
    @abstractmethod
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Process text and convert it to a document.
        
        Args:
            text: Text to process
            metadata: Optional metadata for the document
            
        Returns:
            Processed document
        """
        pass
    
    @abstractmethod
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
        pass


class DocumentChunker(Component):
    """Interface for document chunkers in the ingest pipeline.
    
    Document chunkers split documents into chunks for embedding and retrieval.
    """
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "document_chunker"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "document_chunker"
    
    @abstractmethod
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split a document into chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        pass
    
    @abstractmethod
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
        pass


class EmbeddedDocumentChunk:
    """Class representing an embedded document chunk."""
    
    def __init__(self, chunk: DocumentChunk, embedding: Any):
        """Initialize an embedded document chunk.
        
        Args:
            chunk: The document chunk
            embedding: Vector embedding of the chunk
        """
        self.chunk = chunk
        self.embedding = embedding
    
    @property
    def chunk_id(self) -> str:
        """Get the chunk ID."""
        return self.chunk.chunk_id
    
    @property
    def document_id(self) -> str:
        """Get the document ID."""
        return self.chunk.document_id
    
    @property
    def content(self) -> str:
        """Get the chunk content."""
        return self.chunk.content
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the chunk metadata."""
        return self.chunk.metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation.
        
        Returns:
            Dictionary representation of the embedded chunk
        """
        import numpy as np
        
        result = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata
        }
        
        # Handle different embedding types
        if isinstance(self.embedding, np.ndarray):
            result["embedding"] = self.embedding.tolist()
        elif isinstance(self.embedding, list):
            result["embedding"] = self.embedding
        else:
            result["embedding"] = str(self.embedding)
            
        return result
    
    def __repr__(self) -> str:
        return f"EmbeddedDocumentChunk(id={self.chunk_id}, doc_id={self.document_id})"


class DocumentEmbedder(Component):
    """Interface for document embedders in the ingest pipeline.
    
    Document embedders convert document chunks into vector representations
    that can be used for semantic search and retrieval.
    """
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "document_embedder"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "document_embedder"
    
    @abstractmethod
    def embed_chunks(self, chunks: List[DocumentChunk], 
                    batch_size: Optional[int] = None) -> List[EmbeddedDocumentChunk]:
        """Generate embeddings for a list of document chunks.
        
        Args:
            chunks: List of document chunks to embed
            batch_size: Optional batch size for processing
            
        Returns:
            List of embedded chunks
        """
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> Any:
        """Generate an embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding for the text
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings produced by this embedder.
        
        Returns:
            Dimension of the embeddings
        """
        pass
