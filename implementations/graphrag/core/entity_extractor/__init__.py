"""
Entity Extraction Module for GraphRAG

This module provides components for extracting entities from documents
in the GraphRAG implementation.
"""

from .entity_extractor import EntityExtractor
from .spacy_extractor import SpacyEntityExtractor
from .academic_extractor import AcademicEntityExtractor

__all__ = [
    'EntityExtractor',
    'SpacyEntityExtractor',
    'AcademicEntityExtractor'
]
