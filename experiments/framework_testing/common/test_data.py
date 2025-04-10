"""
Test Data Generation Utilities

This module provides utilities for generating and managing test data
for both PathRAG and GraphRAG frameworks.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

import sys
from pathlib import Path

# Ensure parent directory is in path
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import directly from the ingest module
from ingest.collectors.metadata_schema import DocumentType
from enum import Enum

logger = logging.getLogger(__name__)

class TestDataGenerator:
    """Generator for test data used in framework testing."""
    
    def __init__(self, base_dir: Optional[str] = None, clean_existing: bool = False):
        """
        Initialize the test data generator.
        
        Args:
            base_dir: Base directory for test data (uses temp dir if None)
            clean_existing: Whether to clean existing data in the directory
        """
        if base_dir is None:
            self.base_dir = tempfile.mkdtemp()
            self.temp_created = True
        else:
            self.base_dir = base_dir
            self.temp_created = False
            os.makedirs(self.base_dir, exist_ok=True)
            
            # Clean existing data if requested
            if clean_existing and os.path.exists(self.base_dir):
                for item in os.listdir(self.base_dir):
                    item_path = os.path.join(self.base_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
        
        # Create source document directory
        self.source_dir = os.path.join(self.base_dir, "source_documents")
        os.makedirs(self.source_dir, exist_ok=True)
        
        logger.info(f"Test data will be stored in: {self.base_dir}")
    
    def __del__(self):
        """Clean up temporary directory if created."""
        if hasattr(self, 'temp_created') and self.temp_created and os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
    
    def get_source_dir(self) -> str:
        """Get the source directory path."""
        return self.source_dir
    
    def create_academic_paper(
        self, 
        doc_id: str, 
        title: str, 
        content: str, 
        authors: List[str],
        abstract: str,
        references: Optional[List[Dict[str, Any]]] = None,
        keywords: Optional[List[str]] = None,
        doi: Optional[str] = None,
        journal: Optional[str] = None,
        year: Optional[int] = None,
        categories: Optional[List[str]] = None,
        custom: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an academic paper test document.
        
        Args:
            doc_id: Document ID
            title: Paper title
            content: Paper content
            authors: List of authors
            abstract: Paper abstract
            references: List of references
            keywords: List of keywords
            doi: Digital Object Identifier
            journal: Journal name
            year: Publication year
            categories: List of categories
            custom: Custom metadata
            
        Returns:
            Created document metadata
        """
        return self._create_document(
            doc_id=doc_id,
            doc_type=DocumentType.ACADEMIC_PAPER.value,
            title=title,
            content=content,
            authors=authors,
            abstract=abstract,
            references=references or [],
            keywords=keywords or [],
            doi=doi or f"10.1000/test.{doc_id}",
            journal=journal,
            year=year,
            categories=categories or [],
            custom=custom or {}
        )
    
    def create_documentation(
        self,
        doc_id: str,
        title: str,
        content: str,
        authors: List[str],
        version: Optional[str] = None,
        api_version: Optional[str] = None,
        framework: Optional[str] = None,
        sections: Optional[List[Dict[str, Any]]] = None,
        keywords: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        custom: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a documentation test document.
        
        Args:
            doc_id: Document ID
            title: Documentation title
            content: Documentation content
            authors: List of authors
            version: Documentation version
            api_version: API version
            framework: Framework name
            sections: List of sections
            keywords: List of keywords
            categories: List of categories
            custom: Custom metadata
            
        Returns:
            Created document metadata
        """
        return self._create_document(
            doc_id=doc_id,
            doc_type=DocumentType.DOCUMENTATION.value,
            title=title,
            content=content,
            authors=authors,
            version=version or "1.0.0",
            api_version=api_version,
            framework=framework,
            sections=sections or [],
            keywords=keywords or [],
            categories=categories or [],
            custom=custom or {}
        )
    
    def create_text_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        authors: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        custom: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a general text test document.
        
        Args:
            doc_id: Document ID
            title: Document title
            content: Document content
            authors: List of authors
            keywords: List of keywords
            categories: List of categories
            custom: Custom metadata
            
        Returns:
            Created document metadata
        """
        return self._create_document(
            doc_id=doc_id,
            doc_type="general_text",  # Use string value directly since GENERAL_TEXT might not exist in enum
            title=title,
            content=content,
            authors=authors or [],
            keywords=keywords or [],
            categories=categories or [],
            custom=custom or {}
        )
    
    def _create_document(self, doc_id: str, doc_type: str, title: str, 
                         content: str, authors: List[str], **kwargs) -> Dict[str, Any]:
        """
        Create a document with the given parameters.
        
        Args:
            doc_id: Document ID
            doc_type: Document type
            title: Document title
            content: Document content
            authors: List of authors
            **kwargs: Additional metadata fields
            
        Returns:
            Created document metadata
        """
        # Create document directory
        doc_dir = os.path.join(self.source_dir, doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        # Create paths
        content_path = os.path.join(doc_dir, f"{doc_id}.txt")
        metadata_path = os.path.join(doc_dir, f"{doc_id}.json")
        
        # Create base metadata
        metadata = {
            'doc_id': doc_id,
            'doc_type': doc_type,
            'title': title,
            'authors': authors,
            'content_length': len(content),
            'language': kwargs.get('language', 'en'),
            'storage_path': content_path,
            'metadata_path': metadata_path,
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
            metadata['year'] = kwargs.get('year')
            metadata['references'] = kwargs.get('references', [])
        
        elif doc_type == DocumentType.DOCUMENTATION.value:
            metadata['version'] = kwargs.get('version', '')
            metadata['api_version'] = kwargs.get('api_version', '')
            metadata['framework'] = kwargs.get('framework', '')
        
        # Create content file
        with open(content_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Create metadata file
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata

    def create_sample_dataset(self) -> Dict[str, List[str]]:
        """
        Create a sample dataset with various document types.
        
        Returns:
            Dictionary mapping document types to lists of document IDs
        """
        dataset: Dict[str, List[str]] = {
            'academic_papers': [],
            'documentation': [],
            'text_documents': []
        }
        
        # Create academic papers
        paper1_id = "paper_neural_networks"
        self.create_academic_paper(
            doc_id=paper1_id,
            title="Introduction to Neural Networks",
            content=(
                "Neural networks are a set of algorithms, modeled loosely after the human brain, "
                "that are designed to recognize patterns. They interpret sensory data through a "
                "kind of machine perception, labeling or clustering raw input. The patterns they "
                "recognize are numerical, contained in vectors, into which all real-world data, "
                "be it images, sound, text or time series, must be translated.\n\n"
                "Deep learning is a subset of machine learning where artificial neural networks, "
                "algorithms inspired by the human brain, learn from large amounts of data."
            ),
            authors=["John Smith", "Jane Doe"],
            abstract="This paper provides an introduction to neural networks and their applications.",
            keywords=["neural networks", "deep learning", "machine learning", "artificial intelligence"],
            doi="10.1000/test.01",
            journal="Journal of Artificial Intelligence",
            year=2023
        )
        dataset['academic_papers'].append(paper1_id)
        
        paper2_id = "paper_graph_databases"
        self.create_academic_paper(
            doc_id=paper2_id,
            title="Graph Databases for Knowledge Representation",
            content=(
                "Graph databases are database management systems designed to store, query, and "
                "manipulate graph-structured data following the property graph model. In this model, "
                "entities are represented as nodes (or vertices), and relationships between entities "
                "are represented as edges. Both nodes and edges can have properties associated with them.\n\n"
                "ArangoDB is a multi-model database system that supports graph, document, and key-value "
                "models. It provides a unified query language called AQL (ArangoDB Query Language) "
                "that can be used to query data regardless of the data model."
            ),
            authors=["Alice Johnson", "Bob Williams"],
            abstract="This paper explores the use of graph databases for representing knowledge graphs.",
            keywords=["graph databases", "knowledge representation", "ArangoDB", "knowledge graphs"],
            doi="10.1000/test.02",
            journal="Database Systems Journal",
            year=2024
        )
        dataset['academic_papers'].append(paper2_id)
        
        # Create documentation
        doc1_id = "doc_pathrag"
        self.create_documentation(
            doc_id=doc1_id,
            title="PathRAG Framework Documentation",
            content=(
                "PathRAG is a retrieval-augmented generation framework that utilizes path-based "
                "approaches for knowledge retrieval. It extracts entities and relationships from "
                "documents and forms paths between related entities. These paths can then be used "
                "to provide context for generative AI models.\n\n"
                "The key components of PathRAG include:\n"
                "1. Entity Extractor: Identifies entities in documents\n"
                "2. Relationship Extractor: Identifies relationships between entities\n"
                "3. Path Constructor: Forms paths between entities based on relationships\n"
                "4. Path Vector Store: Stores and retrieves path embeddings\n"
                "5. Query Processor: Processes queries and retrieves relevant paths"
            ),
            authors=["PathRAG Team"],
            version="1.2.0",
            framework="PathRAG",
            sections=[
                {"title": "Introduction", "content": "PathRAG is a retrieval-augmented generation framework..."},
                {"title": "Components", "content": "The key components of PathRAG include..."},
                {"title": "Usage", "content": "To use PathRAG, first initialize the extraction pipeline..."}
            ],
            keywords=["PathRAG", "retrieval-augmented generation", "path-based retrieval"]
        )
        dataset['documentation'].append(doc1_id)
        
        doc2_id = "doc_graphrag"
        self.create_documentation(
            doc_id=doc2_id,
            title="GraphRAG Framework Documentation",
            content=(
                "GraphRAG is a retrieval-augmented generation framework that utilizes graph-based "
                "approaches for knowledge retrieval. It constructs a knowledge graph from documents "
                "by extracting entities and relationships, then uses this graph for retrieval.\n\n"
                "The key components of GraphRAG include:\n"
                "1. Entity Extractor: Identifies entities in documents\n"
                "2. Relationship Extractor: Identifies relationships between entities\n"
                "3. Graph Constructor: Builds a knowledge graph from entities and relationships\n"
                "4. Subgraph Extractor: Extracts relevant subgraphs for queries\n"
                "5. Query Processor: Processes queries and retrieves relevant information"
            ),
            authors=["GraphRAG Team"],
            version="0.9.5",
            framework="GraphRAG",
            sections=[
                {"title": "Introduction", "content": "GraphRAG is a retrieval-augmented generation framework..."},
                {"title": "Components", "content": "The key components of GraphRAG include..."},
                {"title": "Usage", "content": "To use GraphRAG, first initialize the graph constructor..."}
            ],
            keywords=["GraphRAG", "retrieval-augmented generation", "graph-based retrieval"]
        )
        dataset['documentation'].append(doc2_id)
        
        # Create text documents
        text1_id = "text_embedding_models"
        self.create_text_document(
            doc_id=text1_id,
            title="Embedding Models Comparison",
            content=(
                "Text embedding models convert text into numerical vectors that capture semantic meaning. "
                "These vectors can be used for various natural language processing tasks such as semantic "
                "search, clustering, and classification.\n\n"
                "Popular embedding models include:\n"
                "- Sentence-BERT: Optimized for sentence embeddings using BERT\n"
                "- OpenAI Embeddings: High-quality embeddings from OpenAI\n"
                "- E5: Embeddings from Microsoft Research\n"
                "- GTE: General Text Embeddings from LLM based models"
            ),
            authors=["Tech Writer"],
            keywords=["embedding models", "semantic search", "NLP", "vectors"]
        )
        dataset['text_documents'].append(text1_id)
        
        text2_id = "text_arangodb"
        self.create_text_document(
            doc_id=text2_id,
            title="ArangoDB for Vector Search",
            content=(
                "ArangoDB now supports vector search capabilities, making it a powerful tool for "
                "semantic search applications. It combines graph database capabilities with vector "
                "search, enabling complex queries that leverage both graph relationships and semantic similarity.\n\n"
                "Key features of ArangoDB vector search:\n"
                "- Supports multiple vector indexes including HNSW and IVF\n"
                "- Allows filtering based on document attributes\n"
                "- Integrates with graph traversal for relationship-aware search\n"
                "- Provides AQL extensions for vector operations"
            ),
            authors=["Database Expert"],
            keywords=["ArangoDB", "vector search", "semantic search", "graph database"]
        )
        dataset['text_documents'].append(text2_id)
        
        return dataset

# Utility functions to help with testing
def create_test_environment(base_dir: Optional[str] = None) -> Tuple[TestDataGenerator, Dict[str, List[str]]]:
    """
    Create a test environment with sample data.
    
    Args:
        base_dir: Base directory for test data
        
    Returns:
        Tuple of (TestDataGenerator, dataset_info)
    """
    generator = TestDataGenerator(base_dir=base_dir, clean_existing=True)
    dataset: Dict[str, List[str]] = generator.create_sample_dataset()
    
    return generator, dataset
