"""
PathRAG Preprocessors Package.

This package contains preprocessors for PathRAG, including the metadata
preprocessor that transforms universal document metadata into PathRAG-specific formats.
"""

from limnos.implementations.pathrag.preprocessors.metadata_preprocessor import PathRAGMetadataPreprocessor

__all__ = ['PathRAGMetadataPreprocessor']
