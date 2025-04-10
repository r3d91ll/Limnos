"""
Extraction Pipeline for PathRAG

This module provides the extraction pipeline for PathRAG, integrating entity extraction,
relationship extraction, and path construction into a unified workflow.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Set
import networkx as nx
from pathlib import Path

from .entity_extractor import EntityExtractor
from .relationship_extractor import RelationshipExtractor
from .path_constructor import PathConstructor

# Configure logging
logger = logging.getLogger(__name__)

class ExtractionPipeline:
    """
    End-to-end extraction pipeline for PathRAG.
    
    This pipeline orchestrates the process of extracting entities, identifying relationships,
    and constructing paths from document content for path-based retrieval.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the extraction pipeline with configuration.
        
        Args:
            config: Configuration dictionary with options for the entire pipeline
        """
        self.config = config or {}
        
        # Initialize components with their respective configurations
        self.entity_extractor = EntityExtractor(
            self.config.get("entity_extractor", {})
        )
        
        self.relationship_extractor = RelationshipExtractor(
            self.config.get("relationship_extractor", {})
        )
        
        self.path_constructor = PathConstructor(
            self.config.get("path_constructor", {})
        )
        
        # Output directory for path data
        self.output_dir = self.config.get("output_dir", None)
        
        # Whether to save intermediate results
        self.save_intermediates = self.config.get("save_intermediates", False)
        
        # Graph built from processed documents
        self.graph: Optional[nx.MultiDiGraph] = None
        
        # Track processed documents to avoid duplicates
        self.processed_document_ids: Set[str] = set()
        
        # Metadata for all extracted components
        # Define a more specific type for the nested metadata structure
        self.metadata: Dict[str, Any] = {
            "entities": {},      # entity_id -> entity_data
            "relationships": {}, # relationship_id -> relationship_data
            "paths": {},        # path_id -> path_data
            "documents": {},    # document_id -> document_metadata
            "meta": {},         # metadata about this extraction
            "stats": {}         # statistics about extraction results
        }
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document through the extraction pipeline.
        
        Args:
            document: Document dictionary with content and metadata
            
        Returns:
            Dictionary with extraction results
        """
        # Check if document has required fields
        if not document or "content" not in document:
            logger.warning("Document missing required 'content' field")
            return {"error": "Document missing required 'content' field"}
        
        # Check if document has an ID, generate one if not
        if "id" not in document:
            document["id"] = f"doc_{hash(document['content']) % 10000}"
        
        # Skip if already processed
        if document["id"] in self.processed_document_ids:
            logger.info(f"Document {document['id']} already processed, skipping")
            return {"status": "skipped", "document_id": document["id"]}
        
        # Step 1: Extract entities
        logger.info(f"Extracting entities from document {document['id']}")
        entities = self.entity_extractor.extract_document_entities(document)
        
        # Step 2: Extract relationships between entities
        logger.info(f"Extracting relationships from document {document['id']}")
        relationships = self.relationship_extractor.extract_document_relationships(document, entities)
        
        # Save intermediate results if enabled
        if self.save_intermediates and self.output_dir:
            self._save_intermediate_results(document, entities, relationships)
        
        # Update metadata
        self._update_metadata(document, entities, relationships)
        
        # Add to processed documents
        self.processed_document_ids.add(document["id"])
        
        # Return processing results
        return {
            "status": "success",
            "document_id": document["id"],
            "entities_count": len(entities),
            "relationships_count": len(relationships),
            "entities": entities,
            "relationships": relationships
        }
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple documents through the extraction pipeline.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of processing result dictionaries
        """
        results = []
        
        for document in documents:
            result = self.process_document(document)
            results.append(result)
        
        # After processing all documents, build the graph
        self.build_graph()
        
        return results
    
    def build_graph(self) -> nx.MultiDiGraph:
        """
        Build a graph from all processed documents.
        
        Returns:
            NetworkX MultiDiGraph of entities and relationships
        """
        # Collect all entities and relationships
        all_entities = []
        all_relationships = []
        
        for entity_id, entity in self.metadata["entities"].items():
            all_entities.append(entity)
        
        for rel_id, relationship in self.metadata["relationships"].items():
            all_relationships.append(relationship)
        
        logger.info(f"Building graph with {len(all_entities)} entities and {len(all_relationships)} relationships")
        
        # Build the graph
        graph: nx.MultiDiGraph = self.path_constructor.build_graph(all_entities, all_relationships)
        self.graph = graph
        
        return graph
    
    def extract_paths_for_query(self, query: str) -> Dict[str, Any]:
        """
        Extract paths relevant to a query.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with extracted paths and metadata
        """
        if not self.graph:
            logger.warning("Graph not built yet, building now")
            self.build_graph()
        
        # Extract entities from the query
        query_doc = {"id": "query", "content": query}
        query_entities = self.entity_extractor.extract_document_entities(query_doc)
        
        logger.info(f"Extracted {len(query_entities)} entities from query: {query}")
        
        # Construct paths starting from query entities
        paths = self.path_constructor.construct_paths(query_entities, self.graph)
        
        logger.info(f"Constructed {len(paths)} paths for query")
        
        # Update metadata with paths
        for path in paths:
            self.metadata["paths"][path["id"]] = path
        
        # Save paths if output directory is specified
        if self.output_dir:
            self._save_paths(query, paths)
        
        return {
            "query": query,
            "entities": query_entities,
            "paths": paths,
            "paths_count": len(paths)
        }
    
    def _update_metadata(
        self, document: Dict[str, Any], entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]
    ) -> None:
        """
        Update metadata with entities and relationships from a document.
        
        Args:
            document: Document dictionary
            entities: List of extracted entities
            relationships: List of extracted relationships
        """
        # Add document metadata
        self.metadata["documents"][document["id"]] = {
            "id": document["id"],
            "title": document.get("title", ""),
            "path": document.get("path", ""),
            "source": document.get("source", ""),
            "entities_count": len(entities),
            "relationships_count": len(relationships)
        }
        
        # Add entities
        for entity in entities:
            self.metadata["entities"][entity["id"]] = entity
        
        # Add relationships
        for relationship in relationships:
            self.metadata["relationships"][relationship["id"]] = relationship
    
    def _save_intermediate_results(
        self, document: Dict[str, Any], entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]
    ) -> None:
        """
        Save intermediate extraction results to files.
        
        Args:
            document: Document dictionary
            entities: List of extracted entities
            relationships: List of extracted relationships
        """
        if not self.output_dir:
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save entities
        entities_file = os.path.join(
            self.output_dir, f"entities_{document['id']}.json"
        )
        with open(entities_file, "w", encoding="utf-8") as f:
            json.dump(entities, f, indent=2)
        
        # Save relationships
        relationships_file = os.path.join(
            self.output_dir, f"relationships_{document['id']}.json"
        )
        with open(relationships_file, "w", encoding="utf-8") as f:
            json.dump(relationships, f, indent=2)
    
    def _save_paths(self, query: str, paths: List[Dict[str, Any]]) -> None:
        """
        Save paths to a file.
        
        Args:
            query: Query string
            paths: List of path dictionaries
        """
        if not self.output_dir:
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a safe filename from the query
        safe_query = "".join(c if c.isalnum() else "_" for c in query)[:50]
        
        # Save paths
        paths_file = os.path.join(
            self.output_dir, f"paths_{safe_query}.json"
        )
        with open(paths_file, "w", encoding="utf-8") as f:
            json.dump(paths, f, indent=2)
    
    def save_metadata(self, filepath: Optional[str] = None) -> str:
        """
        Save all metadata to a file.
        
        Args:
            filepath: Optional file path to save to
            
        Returns:
            Path to the saved metadata file
        """
        if not filepath and self.output_dir:
            filepath = os.path.join(self.output_dir, "pathrag_metadata.json")
        elif not filepath:
            filepath = "pathrag_metadata.json"
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Add timestamp
        import datetime
        self.metadata["meta"]["timestamp"] = datetime.datetime.now().isoformat()
        
        # Add stats
        self.metadata["stats"] = {
            "documents_count": len(self.metadata["documents"]),
            "entities_count": len(self.metadata["entities"]),
            "relationships_count": len(self.metadata["relationships"]),
            "paths_count": len(self.metadata["paths"])
        }
        
        # Save metadata
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {filepath}")
        
        return filepath
    
    def load_metadata(self, filepath: str) -> Dict[str, Any]:
        """
        Load metadata from a file.
        
        Args:
            filepath: Path to the metadata file
            
        Returns:
            Loaded metadata dictionary
        """
        with open(filepath, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        logger.info(f"Loaded metadata from {filepath}")
        
        # Rebuild the graph from loaded metadata
        self.build_graph()
        
        return self.metadata
    
    def export_graph(self, format: str = "gml", filepath: Optional[str] = None) -> str:
        """
        Export the graph to a file.
        
        Args:
            format: Format to export to (gml, graphml, etc.)
            filepath: Optional file path to save to
            
        Returns:
            Path to the exported graph file
        """
        if not self.graph:
            logger.warning("Graph not built yet, building now")
            self.build_graph()
        
        if not filepath and self.output_dir:
            filepath = os.path.join(self.output_dir, f"pathrag_graph.{format}")
        elif not filepath:
            filepath = f"pathrag_graph.{format}"
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Export graph based on format
        if format == "gml":
            nx.write_gml(self.graph, filepath)
        elif format == "graphml":
            nx.write_graphml(self.graph, filepath)
        elif format == "adjlist":
            nx.write_adjlist(self.graph, filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported graph to {filepath}")
        
        return filepath
    
    def get_document_count(self) -> int:
        """
        Get the number of processed documents.
        
        Returns:
            Number of processed documents
        """
        return len(self.processed_document_ids)
    
    def get_entity_count(self) -> int:
        """
        Get the number of extracted entities.
        
        Returns:
            Number of extracted entities
        """
        return len(self.metadata["entities"])
    
    def get_relationship_count(self) -> int:
        """
        Get the number of extracted relationships.
        
        Returns:
            Number of extracted relationships
        """
        return len(self.metadata["relationships"])
    
    def get_path_count(self) -> int:
        """
        Get the number of constructed paths.
        
        Returns:
            Number of constructed paths
        """
        return len(self.metadata["paths"])
