"""
Universal Document Collector Integration for GraphRAG

This module provides integration components to connect GraphRAG with the
Universal Document Collector, handling document processing, entity extraction,
and graph construction while maintaining the separation between universal and
framework-specific metadata.
"""

import os
import json
import logging
import time
from pathlib import Path as FilePath
from typing import List, Dict, Any, Set, Tuple, Optional, Union, Callable
import shutil
import hashlib
from datetime import datetime
import traceback
import networkx as nx

# Import GraphRAG components
from ...core.entity_extractor.entity_extractor import EntityExtractor
from ...core.relationship_extractor.relationship_extractor import RelationshipExtractor
from ...core.graph_constructor.graph_constructor import GraphConstructor
from ...core.models.entity import Entity
from ...core.models.relationship import Relationship
from ...core.models.document import DocumentReference
from ...preprocessors.metadata_preprocessor import GraphRAGMetadataPreprocessor

# Import Universal Document Collector components
from limnos.ingest.collectors.universal_collector import UniversalDocumentCollector
from limnos.ingest.collectors.metadata_interface import MetadataFormat

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
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the document collector integration with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Source directories (universal document storage)
        self.source_documents_dir = self.config.get(
            'source_documents_dir', 
            os.path.join('/home/todd/ML-Lab/Olympus/limnos/data/source_documents')
        )
        
        # Base directory for GraphRAG-specific data
        self.graphrag_data_dir = self.config.get(
            'graphrag_data_dir',
            os.path.join('/home/todd/ML-Lab/Olympus/limnos/data/implementations/graphrag')
        )
        
        # Directory for processed document metadata
        self.processed_docs_dir = os.path.join(self.graphrag_data_dir, 'metadata')
        
        # Directory for storing graph data
        self.graphs_dir = os.path.join(self.graphrag_data_dir, 'graphs')
        
        # Directory for processing logs
        self.logs_dir = os.path.join(self.graphrag_data_dir, 'logs')
        
        # Create necessary directories
        for directory in [self.processed_docs_dir, self.graphs_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize the Universal Document Collector
        self.universal_collector = self._initialize_universal_collector()
        
        # Initialize the metadata preprocessor
        self.metadata_preprocessor = GraphRAGMetadataPreprocessor(
            output_dir=FilePath(self.graphrag_data_dir)
        )
        
        # Register the metadata preprocessor with the Universal Document Collector
        self._register_metadata_preprocessor()
        
        # Initialize GraphRAG components
        self._initialize_components()
        
        # Track processed documents
        self.processed_document_ids = self._load_processed_document_ids()
        
        # Batch size for processing
        self.batch_size = self.config.get('batch_size', 10)
        
        # Processing timeout per document (seconds)
        self.processing_timeout = self.config.get('processing_timeout', 300)
        
        # Skip document types that aren't supported
        self.supported_extensions = self.config.get('supported_extensions', 
                                                   ['.txt', '.md', '.pdf', '.json'])
        
        # Optional processing hooks
        self.pre_processing_hook = None
        self.post_processing_hook = None
        
        # Redis caching config
        self.use_redis_cache = self.config.get('use_redis_cache', True)
        self.redis_config = {
            'host': self.config.get('redis_host', 'localhost'),
            'port': self.config.get('redis_port', 6379),
            'db': self.config.get('redis_db', 0),
            'ttl': self.config.get('redis_ttl', 3600)
        }
        
        logger.info(f"Initialized GraphRAG Document Collector Integration")
        
    def _initialize_universal_collector(self) -> UniversalDocumentCollector:
        """Initialize the Universal Document Collector.
        
        Returns:
            Initialized UniversalDocumentCollector instance
        """
        collector = UniversalDocumentCollector()
        collector_config = {
            'source_dir': self.source_documents_dir,
            'processor_type': 'academic',  # Default to academic processor
            'processor_config': self.config.get('processor_config', {})
        }
        collector.initialize(collector_config)
        return collector
    
    def _register_metadata_preprocessor(self):
        """Register the GraphRAG metadata preprocessor with the Universal Document Collector."""
        # Register GraphRAG as a framework
        self.universal_collector.register_framework('graphrag')
        
        # Register the metadata preprocessor as an extension point
        self.universal_collector.register_metadata_extension_point(
            'graphrag', 
            self.metadata_preprocessor
        )
        logger.info("Registered GraphRAG metadata preprocessor with Universal Document Collector")
    
    def _initialize_components(self):
        """Initialize the GraphRAG components needed for document processing."""
        # Entity extractor configuration
        entity_extractor_config = self.config.get('entity_extractor_config', {})
        self.entity_extractor = EntityExtractor(entity_extractor_config)
        
        # Relationship extractor configuration
        relationship_extractor_config = self.config.get('relationship_extractor_config', {})
        self.relationship_extractor = RelationshipExtractor(relationship_extractor_config)
        
        # Graph constructor configuration
        graph_constructor_config = self.config.get('graph_constructor_config', {})
        graph_constructor_config.update({
            'graph_dir': self.graphs_dir,
            'use_redis_cache': self.config.get('use_redis_cache', True),
            'redis_host': self.config.get('redis_host', 'localhost'),
            'redis_port': self.config.get('redis_port', 6379),
            'redis_db': self.config.get('redis_db', 0),
            'redis_ttl': self.config.get('redis_ttl', 3600)
        })
        self.graph_constructor = GraphConstructor(graph_constructor_config)
        
    def _load_processed_document_ids(self) -> Set[str]:
        """
        Load the set of document IDs that have already been processed.
        
        Returns:
            Set of document IDs
        """
        processed_index_path = os.path.join(self.processed_docs_dir, 'processed_index.json')
        if os.path.exists(processed_index_path):
            try:
                with open(processed_index_path, 'r') as f:
                    index = json.load(f)
                    return set(index.get('document_ids', []))
            except Exception as e:
                logger.error(f"Error loading processed document index: {e}")
                return set()
        return set()
    
    def _save_processed_document_ids(self):
        """Save the set of processed document IDs."""
        processed_index_path = os.path.join(self.processed_docs_dir, 'processed_index.json')
        try:
            index = {
                'document_ids': list(self.processed_document_ids),
                'last_updated': datetime.now().isoformat()
            }
            with open(processed_index_path, 'w') as f:
                json.dump(index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving processed document index: {e}")
            
    def _get_document_metadata_path(self, document_id: str) -> str:
        """
        Get the path to the universal metadata file for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Path to the metadata JSON file
        """
        return os.path.join(self.source_documents_dir, document_id, f"{document_id}.json")
    
    def _get_framework_metadata_path(self, document_id: str) -> str:
        """
        Get the path to the GraphRAG-specific metadata file for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Path to the framework-specific metadata JSON file
        """
        return os.path.join(self.processed_docs_dir, f"{document_id}.json")
    
    def _get_graph_path(self, document_id: str) -> str:
        """
        Get the path to the graph file for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Path to the graph file
        """
        return os.path.join(self.graphs_dir, f"{document_id}.json")
    
    def process_document(self, document_id: str) -> Dict[str, Any]:
        """
        Process a single document from the Universal Document Collector.
        
        Args:
            document_id: ID of the document to process
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        logger.info(f"Processing document: {document_id}")
        
        result = {
            'document_id': document_id,
            'success': False,
            'error': None,
            'processing_time': 0,
            'entity_count': 0,
            'relationship_count': 0,
            'graph_node_count': 0,
            'graph_edge_count': 0
        }
        
        try:
            # Check if document is already processed
            if document_id in self.processed_document_ids and not self.config.get('force_reprocess', False):
                logger.info(f"Document {document_id} already processed, skipping")
                result['success'] = True
                result['already_processed'] = True
                return result
            
            # Get universal metadata using the metadata provider
            try:
                universal_metadata = self.universal_collector.get_metadata_provider().get_metadata(
                    document_id, MetadataFormat.DICT
                )
            except FileNotFoundError:
                error_msg = f"Universal metadata not found for document {document_id}"
                logger.error(error_msg)
                result['error'] = error_msg
                return result
            
            # Process metadata with GraphRAG preprocessor
            # This will transform universal metadata into GraphRAG-specific format
            self.metadata_preprocessor.update_processing_status(
                document_id, 'entity_extraction', 'in_progress'
            )
            
            graphrag_metadata = self.universal_collector.process_metadata_with_extension(
                document_id, 'graphrag'
            )
            
            # Call pre-processing hook if defined
            if self.pre_processing_hook:
                graphrag_metadata = self.pre_processing_hook(graphrag_metadata)
                
            # Create document reference
            document = DocumentReference(
                document_id=document_id,
                title=universal_metadata.get('title', ''),
                source_path=universal_metadata.get('storage_path', ''),
                metadata=graphrag_metadata
            )
            
            # Extract content for processing
            content = universal_metadata.get('content', '')
            if not content:
                sections = universal_metadata.get('sections', [])
                content = '\n\n'.join([section.get('content', '') for section in sections])
                
            if not content:
                logger.warning(f"No content found for document {document_id}")
                
            # Extract entities using extraction hints from preprocessor
            entity_extraction_config = self.metadata_preprocessor.get_entity_extraction_config(document_id)
            entities = self.entity_extractor.extract_entities(content, document)
            result['entity_count'] = len(entities)
            logger.info(f"Extracted {len(entities)} entities from document {document_id}")
            
            # Update processing status
            self.metadata_preprocessor.update_processing_status(
                document_id, 'entity_extraction', 'completed'
            )
            self.metadata_preprocessor.update_processing_status(
                document_id, 'relationship_extraction', 'in_progress'
            )
            
            # Extract relationships using extraction hints from preprocessor
            relationship_extraction_config = self.metadata_preprocessor.get_relationship_extraction_config(document_id)
            relationships = self.relationship_extractor.extract_relationships(entities, content, document)
            result['relationship_count'] = len(relationships)
            logger.info(f"Extracted {len(relationships)} relationships from document {document_id}")
            
            # Update processing status
            self.metadata_preprocessor.update_processing_status(
                document_id, 'relationship_extraction', 'completed'
            )
            self.metadata_preprocessor.update_processing_status(
                document_id, 'graph_construction', 'in_progress'
            )
            
            # Build graph using configuration from preprocessor
            graph_construction_config = self.metadata_preprocessor.get_graph_construction_config(document_id)
            graph = self.graph_constructor.build_graph(entities, relationships, document)
            result['graph_node_count'] = graph.number_of_nodes()
            result['graph_edge_count'] = graph.number_of_edges()
            
            # Save graph
            graph_path = self._get_graph_path(document_id)
            self.graph_constructor.save_graph(graph, graph_path)
            
            # Update processing status
            self.metadata_preprocessor.update_processing_status(
                document_id, 'graph_construction', 'completed'
            )
            
            # Create and save framework-specific metadata
            framework_metadata = self.metadata_preprocessor.get_framework_metadata(document_id)
            if framework_metadata is None:
                framework_metadata = {}
                
            # Update with processing results
            framework_metadata.update({
                'document_id': document_id,
                'processed_at': datetime.now().isoformat(),
                'entity_count': len(entities),
                'relationship_count': len(relationships),
                'graph_node_count': graph.number_of_nodes(),
                'graph_edge_count': graph.number_of_edges(),
                'graph_path': graph_path,
                'entity_types': self._count_entity_types(entities),
                'relationship_types': self._count_relationship_types(relationships)
            })
            
            # Call post-processing hook if defined
            if self.post_processing_hook:
                framework_metadata = self.post_processing_hook(framework_metadata, universal_metadata)
                
            # Save updated framework-specific metadata
            framework_metadata_path = self._get_framework_metadata_path(document_id)
            with open(framework_metadata_path, 'w') as f:
                json.dump(framework_metadata, f, indent=2)
                
            # Mark document as processed
            self.processed_document_ids.add(document_id)
            self._save_processed_document_ids()
            
            result['success'] = True
            logger.info(f"Successfully processed document {document_id}")
            
        except Exception as e:
            error_msg = f"Error processing document {document_id}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            result['error'] = error_msg
            
            # Update processing status to failed
            try:
                self.metadata_preprocessor.update_processing_status(
                    document_id, 'entity_extraction', 'failed'
                )
                self.metadata_preprocessor.update_processing_status(
                    document_id, 'relationship_extraction', 'failed'
                )
                self.metadata_preprocessor.update_processing_status(
                    document_id, 'graph_construction', 'failed'
                )
            except Exception:
                # Ignore errors when updating status after a failure
                pass
            
        finally:
            result['processing_time'] = time.time() - start_time
            
        return result
    
    def _count_entity_types(self, entities: List[Entity]) -> Dict[str, int]:
        """
        Count the frequency of each entity type.
        
        Args:
            entities: List of entities
            
        Returns:
            Dictionary with entity type counts
        """
        type_counts = {}
        for entity in entities:
            entity_type = entity.entity_type
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        return type_counts
    
    def _count_relationship_types(self, relationships: List[Relationship]) -> Dict[str, int]:
        """
        Count the frequency of each relationship type.
        
        Args:
            relationships: List of relationships
            
        Returns:
            Dictionary with relationship type counts
        """
        type_counts = {}
        for relationship in relationships:
            rel_type = relationship.relationship_type
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
        return type_counts
    
    def process_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """
        Process multiple documents from the Universal Document Collector.
        
        Args:
            document_ids: List of document IDs to process
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        logger.info(f"Batch processing {len(document_ids)} documents")
        
        results = {
            'total_documents': len(document_ids),
            'successful': 0,
            'failed': 0,
            'already_processed': 0,
            'entity_count': 0,
            'relationship_count': 0,
            'graph_node_count': 0,
            'graph_edge_count': 0,
            'processing_time': 0,
            'document_results': {}
        }
        
        # Process in batches
        for i in range(0, len(document_ids), self.batch_size):
            batch = document_ids[i:i+self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1} with {len(batch)} documents")
            
            for doc_id in batch:
                result = self.process_document(doc_id)
                results['document_results'][doc_id] = result
                
                if result.get('success'):
                    results['successful'] += 1
                    if result.get('already_processed'):
                        results['already_processed'] += 1
                    else:
                        results['entity_count'] += result.get('entity_count', 0)
                        results['relationship_count'] += result.get('relationship_count', 0)
                        results['graph_node_count'] += result.get('graph_node_count', 0)
                        results['graph_edge_count'] += result.get('graph_edge_count', 0)
                else:
                    results['failed'] += 1
                    
        results['processing_time'] = time.time() - start_time
        
        # Log summary
        logger.info(f"Batch processing completed in {results['processing_time']:.2f} seconds")
        logger.info(f"Successfully processed {results['successful']} documents")
        logger.info(f"Failed to process {results['failed']} documents")
        
        return results
    
    def get_unprocessed_documents(self) -> List[str]:
        """
        Get list of document IDs that haven't been processed yet.
        
        Returns:
            List of unprocessed document IDs
        """
        # Get all documents from the universal document collector
        all_documents = []
        for filename in os.listdir(self.source_documents_dir):
            if filename.endswith('.json'):
                doc_id = filename.rsplit('.', 1)[0]
                all_documents.append(doc_id)
                
        # Filter for unprocessed documents
        unprocessed = [doc_id for doc_id in all_documents if doc_id not in self.processed_document_ids]
        return unprocessed
    
    def process_all_unprocessed_documents(self) -> Dict[str, Any]:
        """
        Process all documents that haven't been processed yet.
        
        Returns:
            Dictionary with processing results
        """
        unprocessed = self.get_unprocessed_documents()
        logger.info(f"Found {len(unprocessed)} unprocessed documents")
        
        if not unprocessed:
            return {
                'total_documents': 0,
                'successful': 0,
                'failed': 0,
                'message': 'No unprocessed documents found'
            }
            
        return self.process_documents(unprocessed)
    
    def invalidate_document(self, document_id: str) -> bool:
        """
        Invalidate a processed document to force reprocessing.
        
        Args:
            document_id: ID of the document to invalidate
            
        Returns:
            True if invalidation was successful
        """
        if document_id not in self.processed_document_ids:
            logger.warning(f"Document {document_id} is not processed, no need to invalidate")
            return False
            
        # Remove from processed list
        self.processed_document_ids.remove(document_id)
        self._save_processed_document_ids()
        
        # Invalidate cache if using Redis
        if self.use_redis_cache:
            document = DocumentReference(document_id=document_id)
            self.graph_constructor.invalidate_cache(document)
            
        logger.info(f"Invalidated document {document_id}")
        return True
    
    def check_document_status(self, document_id: str) -> Dict[str, Any]:
        """
        Check the processing status of a document.
        
        Args:
            document_id: ID of the document to check
            
        Returns:
            Dictionary with document status
        """
        result = {
            'document_id': document_id,
            'exists': False,
            'processed': False,
            'metadata': {},
            'framework_metadata': {}
        }
        
        # Check if universal metadata exists
        metadata_path = self._get_document_metadata_path(document_id)
        if os.path.exists(metadata_path):
            result['exists'] = True
            try:
                with open(metadata_path, 'r') as f:
                    result['metadata'] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata for document {document_id}: {e}")
                
        # Check if processed
        result['processed'] = document_id in self.processed_document_ids
        
        # Get framework-specific metadata if processed
        if result['processed']:
            framework_metadata_path = self._get_framework_metadata_path(document_id)
            if os.path.exists(framework_metadata_path):
                try:
                    with open(framework_metadata_path, 'r') as f:
                        result['framework_metadata'] = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading framework metadata for document {document_id}: {e}")
                    
        return result
    
    def get_document_entities(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all entities associated with a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            List of entity dictionaries
        """
        if document_id not in self.processed_document_ids:
            logger.warning(f"Document {document_id} is not processed")
            return []
            
        # Load the graph
        graph_path = self._get_graph_path(document_id)
        if not os.path.exists(graph_path):
            logger.warning(f"Graph file not found for document {document_id}")
            return []
            
        # Create document reference
        document = DocumentReference(document_id=document_id)
        
        # Load the graph
        self.graph_constructor.load_graph(graph_path, document)
        
        # Extract entity nodes from the graph
        entities = []
        for node, attrs in self.graph_constructor.graph.nodes(data=True):
            if attrs.get('node_type') == 'entity':
                entity_dict = attrs.copy()
                entity_dict['id'] = node
                entities.append(entity_dict)
                
        return entities
    
    def register_pre_processing_hook(self, hook: Callable):
        """
        Register a pre-processing hook function.
        
        Args:
            hook: Function that takes universal metadata and returns modified metadata
        """
        self.pre_processing_hook = hook
        logger.info("Registered pre-processing hook")
        
    def register_post_processing_hook(self, hook: Callable):
        """
        Register a post-processing hook function.
        
        Args:
            hook: Function that takes framework metadata and universal metadata
                 and returns modified framework metadata
        """
        self.post_processing_hook = hook
        logger.info("Registered post-processing hook")
