"""
Universal Document Collector Integration for GraphRAG (Refactored)

This module provides integration components to connect GraphRAG with the
Universal Document Collector, handling document processing, entity extraction,
and graph construction while maintaining the separation between universal and
framework-specific metadata. This refactored version uses Pydantic models and
the new metadata architecture.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional, Union, Callable
import shutil
import hashlib
from datetime import datetime
import traceback
import networkx as nx

# Import GraphRAG components
from ...core.entity_extractor.entity_extractor import EntityExtractor
from ...core.entity_extractor.spacy_extractor import SpacyEntityExtractor
from ...core.relationship_extractor.relationship_extractor import RelationshipExtractor
from ...core.relationship_extractor.factory import RelationshipExtractorFactory
from ...core.graph_constructor.graph_constructor import GraphConstructor
from ...core.models.entity import Entity
from ...core.models.relationship import Relationship
from ...core.models.document import DocumentReference
from ...preprocessors.graphrag_metadata_preprocessor import GraphRAGMetadataPreprocessor

# Import Universal Document Collector components
from limnos.ingest.collectors.universal_collector_refactored import UniversalDocumentCollector
from limnos.ingest.collectors.metadata_provider import MetadataProvider
from limnos.ingest.collectors.metadata_interface import MetadataFormat
from limnos.ingest.collectors.metadata_factory import MetadataPreprocessorFactory

# Import Pydantic models
from limnos.ingest.models.document import Document
from limnos.ingest.models.metadata import UniversalMetadata, GraphRAGMetadata

# Configure logging
logger = logging.getLogger(__name__)

class DocumentProcessingError(Exception):
    """Exception raised for errors during document processing."""
    pass

class DocumentCollectorIntegration:
    """
    Integration with the Universal Document Collector for GraphRAG.
    
    This component processes documents from the Universal Document Collector,
    extracts entities and relationships, constructs knowledge graphs, and 
    stores them in GraphRAG-specific storage while maintaining proper 
    separation of universal and framework-specific metadata.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the document collector integration with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Base directory for source documents (universal storage)
        self.source_documents_dir = Path(self.config.get(
            'source_documents_dir', 
            '/home/todd/ML-Lab/Olympus/limnos/data/source_documents'
        ))
        
        # Base directory for GraphRAG-specific data
        self.graphrag_data_dir = Path(self.config.get(
            'graphrag_data_dir',
            '/home/todd/ML-Lab/Olympus/limnos/data/implementations/graphrag'
        ))
        
        # Create required directories
        self.graphrag_data_dir.mkdir(parents=True, exist_ok=True)
        (self.graphrag_data_dir / "entities").mkdir(exist_ok=True)
        (self.graphrag_data_dir / "relationships").mkdir(exist_ok=True)
        (self.graphrag_data_dir / "graphs").mkdir(exist_ok=True)
        (self.graphrag_data_dir / "logs").mkdir(exist_ok=True)
        
        # Initialize the Universal Document Collector
        self.universal_collector = self._initialize_universal_collector()
        
        # Initialize GraphRAG components
        self._initialize_components()
        
        # Track processed documents
        self.processed_document_ids = self._load_processed_document_ids()
    
    def _initialize_universal_collector(self) -> UniversalDocumentCollector:
        """Initialize the Universal Document Collector."""
        collector = UniversalDocumentCollector()
        collector_config = {
            'source_dir': str(self.source_documents_dir),
            'processor_type': 'academic'
        }
        collector.initialize(collector_config)
        
        # Register GraphRAG metadata preprocessor
        self._register_metadata_preprocessor(collector)
        
        return collector
    
    def _register_metadata_preprocessor(self, collector: UniversalDocumentCollector) -> None:
        """
        Register the GraphRAG metadata preprocessor with the Universal Document Collector.
        
        Args:
            collector: Universal Document Collector instance
        """
        # Using the factory to create the preprocessor
        graphrag_preprocessor = MetadataPreprocessorFactory.create_preprocessor(
            'graphrag',
            {'output_dir': str(self.graphrag_data_dir)}
        )
        
        # Register with the collector
        collector.register_extension_point('graphrag', graphrag_preprocessor)
        
        logger.info("GraphRAG metadata preprocessor registered with the Universal Document Collector")
    
    def _initialize_components(self) -> None:
        """Initialize GraphRAG components."""
        # Initialize entity extractor with a concrete implementation
        entity_extractor_config = self.config.get('entity_extractor_config', {})
        self.entity_extractor = SpacyEntityExtractor(entity_extractor_config)
        
        # Initialize relationship extractor with a concrete implementation
        relationship_extractor_config = self.config.get('relationship_extractor_config', {})
        # Use the factory to create a concrete implementation (composite extractor with appropriate extractors)
        document_type = self.config.get('document_type', 'text')
        self.relationship_extractor = RelationshipExtractorFactory.create_composite_extractor(
            document_type, relationship_extractor_config)
        
        # Initialize graph constructor
        self.graph_constructor = GraphConstructor(self.config.get('graph_constructor_config', {}))
    
    def _load_processed_document_ids(self) -> Set[str]:
        """
        Load the set of document IDs that have been processed.
        
        Returns:
            Set of processed document IDs
        """
        processed_ids_file = self.graphrag_data_dir / "processed_documents.json"
        
        if processed_ids_file.exists():
            try:
                with open(processed_ids_file, 'r') as f:
                    return set(json.load(f))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading processed document IDs: {e}")
                return set()
        else:
            return set()
    
    def _save_processed_document_ids(self) -> None:
        """Save the set of processed document IDs."""
        processed_ids_file = self.graphrag_data_dir / "processed_documents.json"
        
        try:
            with open(processed_ids_file, 'w') as f:
                json.dump(list(self.processed_document_ids), f, indent=2)
        except IOError as e:
            logger.error(f"Error saving processed document IDs: {e}")
    
    def process_document(self, doc_id: str) -> bool:
        """
        Process a document with the GraphRAG pipeline.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if processing was successful, False otherwise
        """
        logger.info(f"Processing document: {doc_id}")
        
        try:
            # Get document from the Universal Document Collector
            doc_result = self.universal_collector.get_document(doc_id)
            if not doc_result:
                logger.error(f"Document not found: {doc_id}")
                return False
            
            document, file_paths = doc_result
            
            # Get GraphRAG-specific metadata
            graphrag_metadata = self.universal_collector.get_framework_metadata(doc_id, 'graphrag')
            if not graphrag_metadata:
                logger.error(f"GraphRAG metadata not found for document: {doc_id}")
                return False
            
            # Extract entities
            entities = self._extract_entities(document, graphrag_metadata)
            
            # Extract relationships
            relationships = self._extract_relationships(document, entities, graphrag_metadata)
            
            # Construct graph
            graph = self._construct_graph(document, entities, relationships, graphrag_metadata)
            
            # Mark document as processed
            self.processed_document_ids.add(doc_id)
            self._save_processed_document_ids()
            
            # Update processing status
            extension_point = self.universal_collector._metadata_provider.get_extension_point('graphrag')
            if extension_point:
                extension_point.update_processing_status(doc_id, {
                    'metadata_processed': True,
                    'entities_extracted': True,
                    'relationships_extracted': True,
                    'graph_constructed': True
                })
            
            logger.info(f"Document {doc_id} processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _extract_entities(self, document: Document, graphrag_metadata: Dict[str, Any]) -> List[Entity]:
        """
        Extract entities from a document.
        
        Args:
            document: Document
            graphrag_metadata: GraphRAG-specific metadata
            
        Returns:
            List of extracted entities
        """
        logger.info(f"Extracting entities from document: {document.doc_id}")
        
        # Get entity extraction configuration
        if 'entity_extraction_config' in graphrag_metadata:
            config = graphrag_metadata['entity_extraction_config']
        else:
            # Fallback for older metadata format
            extension_point = self.universal_collector._metadata_provider.get_extension_point('graphrag')
            if extension_point:
                config = extension_point.get_entity_extraction_config(document.doc_id)
            else:
                config = {}
        
        # Extract entities
        # Create metadata dictionary with document_id and other config items
        metadata = {
            "document_id": document.doc_id,
            **config  # Include any additional config items
        }
        # Call with the correct signature: extract_entities(text, metadata)
        entities = self.entity_extractor.extract_entities(
            text=document.content,
            metadata=metadata
        )
        
        # Save entities to file
        entities_path = graphrag_metadata.get('entities_path', 
                                             str(self.graphrag_data_dir / "entities" / f"{document.doc_id}.json"))
        entities_serializable = [entity.to_dict() for entity in entities]
        
        with open(entities_path, 'w') as f:
            json.dump(entities_serializable, f, indent=2)
        
        logger.info(f"Extracted {len(entities)} entities from document {document.doc_id}")
        return entities
    
    def _extract_relationships(self, document: Document, entities: List[Entity], 
                               graphrag_metadata: Dict[str, Any]) -> List[Relationship]:
        """
        Extract relationships from a document.
        
        Args:
            document: Document
            entities: List of entities
            graphrag_metadata: GraphRAG-specific metadata
            
        Returns:
            List of extracted relationships
        """
        logger.info(f"Extracting relationships from document: {document.doc_id}")
        
        # Get relationship extraction configuration
        if 'relationship_extraction_config' in graphrag_metadata:
            config = graphrag_metadata['relationship_extraction_config']
        else:
            # Fallback for older metadata format
            extension_point = self.universal_collector._metadata_provider.get_extension_point('graphrag')
            if extension_point:
                config = extension_point.get_relationship_extraction_config(document.doc_id)
            else:
                config = {}
        
        # Extract relationships
        # Create metadata dictionary with document_id and other config items
        metadata = {
            "document_id": document.doc_id,
            **config  # Include any additional config items
        }
        # Call with the correct signature: extract_relationships(text, entities, metadata)
        relationships = self.relationship_extractor.extract_relationships(
            text=document.content,
            entities=entities,
            metadata=metadata
        )
        
        # Save relationships to file
        relationships_path = graphrag_metadata.get('relationships_path', 
                                                str(self.graphrag_data_dir / "relationships" / f"{document.doc_id}.json"))
        relationships_serializable = [rel.to_dict() for rel in relationships]
        
        with open(relationships_path, 'w') as f:
            json.dump(relationships_serializable, f, indent=2)
        
        logger.info(f"Extracted {len(relationships)} relationships from document {document.doc_id}")
        return relationships
    
    def _construct_graph(self, document: Document, entities: List[Entity], 
                         relationships: List[Relationship], graphrag_metadata: Dict[str, Any]) -> nx.Graph:
        """
        Construct a knowledge graph from entities and relationships.
        
        Args:
            document: Document
            entities: List of entities
            relationships: List of relationships
            graphrag_metadata: GraphRAG-specific metadata
            
        Returns:
            Constructed knowledge graph
        """
        logger.info(f"Constructing graph for document: {document.doc_id}")
        
        # Get graph construction configuration
        if 'graph_construction_config' in graphrag_metadata:
            config = graphrag_metadata['graph_construction_config']
        else:
            # Fallback for older metadata format
            extension_point = self.universal_collector._metadata_provider.get_extension_point('graphrag')
            if extension_point:
                config = extension_point.get_graph_construction_config(document.doc_id)
            else:
                config = {}
        
        # Create document reference
        # Get title with a fallback to "Unknown" and ensure it's a string
        doc_title = str(document.metadata.title) if hasattr(document.metadata, 'title') and document.metadata.title is not None else "Unknown"
        
        # Create document reference with correct parameter names
        doc_ref = DocumentReference(
            document_id=document.doc_id,
            title=doc_title,
            document_type=document.metadata.doc_type.value if hasattr(document.metadata, 'doc_type') else "text",
            metadata={
                'doc_id': document.doc_id,
                'title': doc_title,
                'doc_type': document.metadata.doc_type.value if hasattr(document.metadata, 'doc_type') else "unknown"
            }
        )
        
        # Construct graph - use build_graph method which is the correct method name
        # The method does not take a config parameter, it uses the class's config
        graph = self.graph_constructor.build_graph(
            document=doc_ref,
            entities=entities,
            relationships=relationships
        )
        
        # Save graph to file
        graph_path = graphrag_metadata.get('graph_path', 
                                          str(self.graphrag_data_dir / "graphs" / f"{document.doc_id}.json"))
        
        # Convert graph to serializable format
        graph_data = nx.node_link_data(graph)
        
        with open(graph_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        logger.info(f"Constructed graph for document {document.doc_id} with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        # Explicit cast to nx.Graph to satisfy type checking
        return graph
    
    def process_all_documents(self) -> Tuple[int, int]:
        """
        Process all documents that haven't been processed yet.
        
        Returns:
            Tuple of (number of documents processed, number of documents with errors)
        """
        # Get all document IDs from the Universal Document Collector
        all_doc_ids = set(self.universal_collector.list_documents())
        
        # Filter out already processed documents
        unprocessed_doc_ids = all_doc_ids - self.processed_document_ids
        
        logger.info(f"Found {len(unprocessed_doc_ids)} unprocessed documents")
        
        success_count = 0
        error_count = 0
        
        for doc_id in unprocessed_doc_ids:
            if self.process_document(doc_id):
                success_count += 1
            else:
                error_count += 1
        
        return success_count, error_count
    
    def get_document_processing_status(self, doc_id: str) -> Dict[str, Any]:
        """
        Get the processing status of a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Status dictionary
        """
        # Check if document exists
        if not self.universal_collector.get_document(doc_id):
            return {'exists': False}
        
        # Get processing status from GraphRAG extension point
        status = self.universal_collector.get_processing_status(doc_id, 'graphrag')
        
        # Add additional status information
        status['in_processed_list'] = doc_id in self.processed_document_ids
        
        return status
    
    def reprocess_document(self, doc_id: str, force: bool = False) -> bool:
        """
        Reprocess a document that has already been processed.
        
        Args:
            doc_id: Document ID
            force: If True, reprocess even if document is already in processed list
            
        Returns:
            True if reprocessing was successful, False otherwise
        """
        # Skip if document is already processed and force is False
        if doc_id in self.processed_document_ids and not force:
            logger.info(f"Document {doc_id} already processed, skipping")
            return False
        
        # Remove from processed documents list to allow reprocessing
        if doc_id in self.processed_document_ids:
            self.processed_document_ids.remove(doc_id)
            self._save_processed_document_ids()
        
        # Reprocess
        return self.process_document(doc_id)
    
    def get_document_graph(self, doc_id: str) -> Optional[nx.Graph]:
        """
        Get the knowledge graph for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Knowledge graph if available, None otherwise
        """
        # Check if document has been processed
        if doc_id not in self.processed_document_ids:
            logger.warning(f"Document {doc_id} has not been processed yet")
            return None
        
        # Get GraphRAG-specific metadata
        graphrag_metadata = self.universal_collector.get_framework_metadata(doc_id, 'graphrag')
        if not graphrag_metadata:
            logger.error(f"GraphRAG metadata not found for document: {doc_id}")
            return None
        
        # Get graph path
        graph_path = graphrag_metadata.get('graph_path', 
                                          str(self.graphrag_data_dir / "graphs" / f"{doc_id}.json"))
        
        if not Path(graph_path).exists():
            logger.error(f"Graph file not found for document: {doc_id}")
            return None
        
        # Load graph from file
        try:
            with open(graph_path, 'r') as f:
                graph_data = json.load(f)
            
            graph = nx.node_link_graph(graph_data)
            # Cast explicitly as nx.Graph to satisfy the type checker
            result_graph: nx.Graph = graph
            return result_graph
        except Exception as e:
            logger.error(f"Error loading graph for document {doc_id}: {e}")
            return None
