"""
Metadata Factory for Limnos.

This module provides factory functions for creating and registering
metadata preprocessors for different RAG frameworks.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List

from limnos.ingest.collectors.metadata_interface import MetadataExtensionPoint
from limnos.ingest.collectors.metadata_provider import MetadataProvider
from limnos.implementations.graphrag.preprocessors.graphrag_metadata_preprocessor import GraphRAGMetadataPreprocessor
from limnos.implementations.pathrag.preprocessors.pathrag_metadata_preprocessor import PathRAGMetadataPreprocessor


class MetadataPreprocessorFactory:
    """Factory for creating metadata preprocessors for different RAG frameworks."""
    
    @staticmethod
    def create_preprocessor(framework: str, config: Optional[Dict[str, Any]] = None) -> MetadataExtensionPoint:
        """Create a metadata preprocessor for a specific framework.
        
        Args:
            framework: Name of the framework (e.g., 'graphrag', 'pathrag')
            config: Optional configuration dictionary
            
        Returns:
            Initialized metadata preprocessor
            
        Raises:
            ValueError: If framework is not supported
        """
        config = config or {}
        
        if framework.lower() == 'graphrag':
            output_dir = config.get('output_dir', None)
            if output_dir:
                output_dir = Path(output_dir)
            return GraphRAGMetadataPreprocessor(output_dir=output_dir)
            
        elif framework.lower() == 'pathrag':
            output_dir = config.get('output_dir', None)
            if output_dir:
                output_dir = Path(output_dir)
            return PathRAGMetadataPreprocessor(output_dir=output_dir)
            
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    @staticmethod
    def register_all_preprocessors(metadata_provider: MetadataProvider, 
                                  frameworks: Optional[List[str]] = None,
                                  config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Register all metadata preprocessors with a metadata provider.
        
        Args:
            metadata_provider: Metadata provider to register with
            frameworks: Optional list of framework names to register
                        (if None, registers all supported frameworks)
            config: Optional configuration dictionary
            
        Returns:
            List of registered framework names
        """
        config = config or {}
        
        # Default to all supported frameworks if none specified
        if not frameworks:
            frameworks = ['graphrag', 'pathrag']
        
        registered = []
        
        for framework in frameworks:
            try:
                preprocessor = MetadataPreprocessorFactory.create_preprocessor(framework, config)
                metadata_provider.register_extension_point(framework, preprocessor)
                registered.append(framework)
            except ValueError as e:
                # Log error but continue with other frameworks
                import logging
                logging.getLogger(__name__).warning(f"Failed to register {framework}: {e}")
        
        return registered
