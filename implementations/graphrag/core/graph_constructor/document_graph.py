"""
Document Graph for GraphRAG

This module implements the DocumentGraph class that manages document-specific
graph operations in the GraphRAG implementation.
"""

import networkx as nx
from typing import Dict, List, Optional, Any, Set, Tuple, Union
import logging
import os
from datetime import datetime
import json

from ..models.entity import Entity
from ..models.relationship import Relationship
from ..models.document import DocumentReference
from .graph_constructor import GraphConstructor


class DocumentGraph:
    """
    Class for managing document-specific graphs in GraphRAG.
    
    The DocumentGraph creates and manages graph representations of individual
    documents, including their entities, relationships, and structure. It provides
    methods for building, updating, and querying document graphs.
    """
    
    def __init__(self, document: DocumentReference, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the document graph.
        
        Args:
            document: Document reference for this graph
            config: Configuration dictionary with options for graph construction
        """
        self.document = document
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize graph constructor
        self.graph_constructor = GraphConstructor(self.config)
        
        # Set up document-specific graph directory
        self.graph_dir = os.path.join(
            self.graph_constructor.graph_dir,
            "document_graphs",
            document.document_id
        )
        os.makedirs(self.graph_dir, exist_ok=True)
        
        # Initialize document graph
        self.graph = None
        self.entities = []
        self.relationships = []
        
        self.logger.info(f"Initialized DocumentGraph for document: {document.document_id}")
    
    def build_graph(self, entities: List[Entity], relationships: List[Relationship],
                   sections: Optional[List[Dict[str, Any]]] = None) -> nx.Graph:
        """
        Build a graph for this document from entities and relationships.
        
        Args:
            entities: List of entities from the document
            relationships: List of relationships from the document
            sections: Optional list of document sections
            
        Returns:
            The constructed document graph
        """
        self.logger.info(f"Building document graph for {self.document.document_id} with "
                        f"{len(entities)} entities and {len(relationships)} relationships")
        
        # Store entities and relationships
        self.entities = entities
        self.relationships = relationships
        
        # Build the graph
        self.graph = self.graph_constructor.build_graph(entities, relationships, self.document)
        
        # Add section nodes if available
        if sections and self.graph_constructor.include_section_nodes:
            self.graph_constructor.add_section_nodes(self.document, sections, entities)
            
        # Add document metadata to graph
        self.graph.graph["document_id"] = self.document.document_id
        self.graph.graph["document_title"] = self.document.title
        self.graph.graph["document_type"] = self.document.document_type
        
        self.logger.info(f"Built document graph with {self.graph.number_of_nodes()} nodes "
                        f"and {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def update_graph(self, new_entities: Optional[List[Entity]] = None,
                    new_relationships: Optional[List[Relationship]] = None,
                    sections: Optional[List[Dict[str, Any]]] = None) -> nx.Graph:
        """
        Update the document graph with new entities and relationships.
        
        Args:
            new_entities: New entities to add to the graph
            new_relationships: New relationships to add to the graph
            sections: Updated document sections
            
        Returns:
            The updated document graph
        """
        if self.graph is None:
            if new_entities and new_relationships:
                return self.build_graph(new_entities, new_relationships, sections)
            else:
                self.logger.error("Cannot update graph: graph not initialized and no entities/relationships provided")
                raise ValueError("Graph not initialized and no entities/relationships provided")
        
        self.logger.info(f"Updating document graph for {self.document.document_id}")
        
        # Add new entities
        if new_entities:
            for entity in new_entities:
                if entity not in self.entities:
                    self.graph_constructor.add_entity(entity)
                    self.entities.append(entity)
            
        # Add new relationships
        if new_relationships:
            for relationship in new_relationships:
                if relationship not in self.relationships:
                    self.graph_constructor.add_relationship(relationship)
                    self.relationships.append(relationship)
                    
        # Update sections if provided
        if sections and self.graph_constructor.include_section_nodes:
            # Remove existing section nodes
            section_nodes = [n for n, attrs in self.graph.nodes(data=True) 
                            if attrs.get("node_type") == "section" and 
                            attrs.get("document_id") == self.document.document_id]
            self.graph.remove_nodes_from(section_nodes)
            
            # Add updated section nodes
            self.graph_constructor.add_section_nodes(self.document, sections, self.entities)
            
        self.logger.info(f"Updated document graph: now has {self.graph.number_of_nodes()} nodes "
                        f"and {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def save_graph(self, filename: Optional[str] = None) -> str:
        """
        Save the document graph to disk.
        
        Args:
            filename: Optional filename to save to (default: auto-generated)
            
        Returns:
            Path to the saved graph file
        """
        if self.graph is None:
            self.logger.error("Cannot save graph: graph not initialized")
            raise ValueError("Graph not initialized")
            
        if not filename:
            # Generate filename based on document ID and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.document.document_id}_{timestamp}.pkl"
            
        # Save using the graph constructor
        filepath = os.path.join(self.graph_dir, filename)
        self.graph_constructor.graph = self.graph  # Set the graph to save
        saved_path = self.graph_constructor.save_graph(filepath)
        
        # Save metadata
        metadata_path = os.path.join(self.graph_dir, f"{os.path.splitext(filename)[0]}_metadata.json")
        self._save_metadata(metadata_path)
        
        return saved_path
    
    def load_graph(self, filepath: str) -> nx.Graph:
        """
        Load a document graph from disk.
        
        Args:
            filepath: Path to the graph file
            
        Returns:
            The loaded document graph
        """
        # Load using the graph constructor
        self.graph = self.graph_constructor.load_graph(filepath)
        
        # Verify this is the correct document graph
        graph_doc_id = self.graph.graph.get("document_id")
        if graph_doc_id != self.document.document_id:
            self.logger.warning(f"Loaded graph has document ID {graph_doc_id}, "
                              f"but expected {self.document.document_id}")
            
        # Extract entities and relationships from the graph
        self._extract_entities_and_relationships()
        
        return self.graph
    
    def _extract_entities_and_relationships(self) -> None:
        """
        Extract entities and relationships from the loaded graph.
        """
        if self.graph is None:
            return
            
        # Clear existing lists
        self.entities = []
        self.relationships = []
        
        # Extract entities from nodes
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") not in ["document", "section"]:
                # Reconstruct entity from node attributes
                entity_data = {
                    "id": node,
                    "text": attrs.get("label", ""),
                    "type": attrs.get("entity_type", "unknown"),
                    "source_document_id": self.document.document_id,
                    "confidence": attrs.get("confidence", 1.0),
                    "positions": attrs.get("positions", [])
                }
                self.entities.append(Entity(**entity_data))
                
        # Extract relationships from edges
        for u, v, attrs in self.graph.edges(data=True):
            if attrs.get("edge_type") not in ["contains", "has_section", "has_subsection"]:
                # Reconstruct relationship from edge attributes
                rel_data = {
                    "source_id": u,
                    "target_id": v,
                    "type": attrs.get("edge_type", "unknown"),
                    "source_document_id": self.document.document_id,
                    "confidence": attrs.get("weight", 1.0),
                    "bidirectional": attrs.get("bidirectional", False),
                    "metadata": {k: v for k, v in attrs.items() 
                               if k not in ["edge_type", "weight", "bidirectional"]}
                }
                self.relationships.append(Relationship(**rel_data))
    
    def _save_metadata(self, metadata_path: str) -> None:
        """
        Save document graph metadata to a JSON file.
        
        Args:
            metadata_path: Path for the metadata file
        """
        if self.graph is None:
            return
            
        # Collect metadata
        metadata = {
            "document_id": self.document.document_id,
            "document_title": self.document.title,
            "document_type": self.document.document_type,
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
            "created_at": self.graph.graph.get("created_at", datetime.now().isoformat()),
            "entity_types": {},
            "relationship_types": {}
        }
        
        # Count entity types
        for entity in self.entities:
            entity_type = entity.type
            metadata["entity_types"][entity_type] = metadata["entity_types"].get(entity_type, 0) + 1
            
        # Count relationship types
        for relationship in self.relationships:
            rel_type = relationship.type
            metadata["relationship_types"][rel_type] = metadata["relationship_types"].get(rel_type, 0) + 1
            
        # Save metadata
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"Saved document graph metadata to {metadata_path}")
        except Exception as e:
            self.logger.error(f"Failed to save document graph metadata: {str(e)}")
    
    def get_entity_subgraph(self, entity_ids: List[str], 
                          include_neighbors: bool = True,
                          max_distance: int = 1) -> nx.Graph:
        """
        Extract a subgraph containing the specified entities.
        
        Args:
            entity_ids: List of entity IDs to include
            include_neighbors: Whether to include neighboring entities
            max_distance: Maximum distance for neighbor inclusion
            
        Returns:
            Subgraph containing the specified entities and their neighbors
        """
        if self.graph is None:
            self.logger.error("Cannot extract subgraph: graph not initialized")
            raise ValueError("Graph not initialized")
            
        # Start with the specified entities
        nodes_to_include = set(entity_ids)
        
        # Add neighbors if requested
        if include_neighbors and max_distance > 0:
            for entity_id in entity_ids:
                if self.graph.has_node(entity_id):
                    # Get neighbors within max_distance
                    for node, distance in nx.single_source_shortest_path_length(
                        self.graph, entity_id, cutoff=max_distance).items():
                        if distance > 0:  # Don't include the source node again
                            nodes_to_include.add(node)
                            
        # Create the subgraph
        subgraph = self.graph.subgraph(nodes_to_include).copy()
        
        return subgraph
    
    def get_document_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the document graph.
        
        Returns:
            Dictionary with document graph summary information
        """
        if self.graph is None:
            return {
                "document_id": self.document.document_id,
                "document_title": self.document.title,
                "status": "not_initialized"
            }
            
        # Get graph statistics
        stats = self.graph_constructor.get_graph_stats()
        
        # Add document-specific information
        summary = {
            "document_id": self.document.document_id,
            "document_title": self.document.title,
            "document_type": self.document.document_type,
            "status": "initialized",
            "node_count": stats["node_count"],
            "edge_count": stats["edge_count"],
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
            "entity_types": {},
            "relationship_types": {}
        }
        
        # Count entity types
        for entity in self.entities:
            entity_type = entity.type
            summary["entity_types"][entity_type] = summary["entity_types"].get(entity_type, 0) + 1
            
        # Count relationship types
        for relationship in self.relationships:
            rel_type = relationship.type
            summary["relationship_types"][rel_type] = summary["relationship_types"].get(rel_type, 0) + 1
            
        return summary
