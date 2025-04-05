"""
GraphRAG Preprocessors Package.

This package contains preprocessors for GraphRAG, including the metadata
preprocessor that transforms universal document metadata into GraphRAG-specific formats.
"""

from limnos.implementations.graphrag.preprocessors.metadata_preprocessor import GraphRAGMetadataPreprocessor

__all__ = ['GraphRAGMetadataPreprocessor']
