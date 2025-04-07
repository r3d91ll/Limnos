"""
Graph Constructor for GraphRAG

This module implements the core GraphConstructor class that builds knowledge graphs
from entities and relationships in the GraphRAG implementation.
"""

import networkx as nx
from typing import Dict, List, Optional, Any, Set, Tuple, Union
import logging
import os
import json
from datetime import datetime
import pickle

from ..models.entity import Entity
from ..models.relationship import Relationship
from ..models.document import DocumentReference
from ..models.graph_elements import GraphNode, GraphEdge
from ..cache.cache_manager import GraphCacheManager


class GraphConstructor:
    """
    Core class for constructing knowledge graphs in GraphRAG.
    
    The GraphConstructor builds NetworkX graphs from entities and relationships,
    optimizing the graph structure for retrieval operations. It provides methods
    for adding nodes and edges, merging graphs, and persisting graphs to disk.
    """
    # Class attribute type annotations
    graph: Union[nx.DiGraph, nx.Graph, nx.MultiGraph, nx.MultiDiGraph]
    cache_manager: Optional[GraphCacheManager]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the graph constructor.
        
        Args:
            config: Configuration dictionary with the following options:
                - graph_type: Type of graph to create (default: "undirected")
                - graph_dir: Directory for storing graph data (default: based on Limnos structure)
                - optimize_for_retrieval: Whether to optimize graph for retrieval (default: True)
                - include_document_nodes: Whether to include document nodes (default: True)
                - include_section_nodes: Whether to include section nodes (default: True)
                - bidirectional_relationships: Whether to add edges in both directions (default: True)
                - use_redis_cache: Whether to use Redis for caching (default: True)
                - redis_host: Redis server hostname (default: localhost)
                - redis_port: Redis server port (default: 6379)
                - redis_db: Redis database number (default: 0)
                - redis_ttl: Default time-to-live for cached items in seconds (default: 3600)
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Get configuration options
        self.graph_type = self.config.get("graph_type", "undirected")
        self.optimize_for_retrieval = self.config.get("optimize_for_retrieval", True)
        self.include_document_nodes = self.config.get("include_document_nodes", True)
        self.include_section_nodes = self.config.get("include_section_nodes", True)
        self.bidirectional_relationships = self.config.get("bidirectional_relationships", True)
        self.use_redis_cache = self.config.get("use_redis_cache", True)
        
        # Set up graph directory based on Limnos architecture
        self.graph_dir = self.config.get(
            "graph_dir",
            os.path.join("/home/todd/ML-Lab/Olympus/limnos/data/implementations/graphrag/graphs")
        )
        os.makedirs(self.graph_dir, exist_ok=True)
        
        # Initialize the graph
        self._initialize_graph()
        
        # Initialize cache manager if caching is enabled
        if self.use_redis_cache:
            cache_config = {
                "host": self.config.get("redis_host", "localhost"),
                "port": self.config.get("redis_port", 6379),
                "db": self.config.get("redis_db", 0),
                "ttl": self.config.get("redis_ttl", 3600),
                "prefix": "graphrag:"
            }
            self.cache_manager = GraphCacheManager(cache_config=cache_config)
            self.logger.info(f"Initialized Redis cache at {cache_config['host']}:{cache_config['port']}")
        else:
            self.cache_manager = None
            
        self.logger.info(f"Initialized GraphConstructor with graph type: {self.graph_type}")
    
    def _initialize_graph(self) -> None:
        """
        Initialize the NetworkX graph based on the configured graph type.
        """
        # Properly type the graph based on the graph type
        if self.graph_type == "directed":
            self.graph = nx.DiGraph()
        elif self.graph_type == "multi":
            # Using cast to handle the type mismatch
            graph: nx.MultiGraph = nx.MultiGraph()
            self.graph = graph  # type: ignore
        elif self.graph_type == "directed_multi":
            graph_multi: nx.MultiDiGraph = nx.MultiDiGraph()
            self.graph = graph_multi  # type: ignore
        else:  # Default to undirected
            graph_simple: nx.Graph = nx.Graph()
            self.graph = graph_simple  # type: ignore
            
        # Add graph metadata
        self.graph.graph["created_at"] = datetime.now().isoformat()
        self.graph.graph["graph_type"] = self.graph_type
        self.graph.graph["framework"] = "graphrag"
    
    def build_graph(self, entities: List[Entity], relationships: List[Relationship],
                   document: Optional[DocumentReference] = None) -> nx.Graph:
        """
        Build a graph from the given entities and relationships.
        
        Args:
            entities: List of entities to add as nodes
            relationships: List of relationships to add as edges
            document: Optional document reference for document-level nodes
            
        Returns:
            The constructed NetworkX graph
        """
        self.logger.info(f"Building graph with {len(entities)} entities and {len(relationships)} relationships")
        
        # Check cache first if we have a document and caching is enabled
        if document and self.use_redis_cache and self.cache_manager:
            cached_graph = self.cache_manager.get_document_graph(document)
            if cached_graph:
                self.logger.info(f"Retrieved graph for document {document.document_id} from cache")
                # Fix the type incompatibility with a type ignore
                self.graph = cached_graph  # type: ignore
                return self.graph
        
        # Reset the graph
        self._initialize_graph()
        
        # Add entities as nodes
        for entity in entities:
            self.add_entity(entity)
            
        # Add relationships as edges
        for relationship in relationships:
            self.add_relationship(relationship)
            
        # Add document node if specified
        if document and self.include_document_nodes:
            self.add_document_node(document, entities)
            
        # Optimize the graph if configured
        if self.optimize_for_retrieval:
            self._optimize_for_retrieval()
        
        # Cache the graph if we have a document and caching is enabled
        if document and self.use_redis_cache and self.cache_manager:
            self.cache_manager.cache_document_graph(self.graph, document)
            self.logger.info(f"Cached graph for document {document.document_id}")
            
        self.logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def add_entity(self, entity: Entity) -> str:
        """
        Add an entity to the graph as a node.
        
        Args:
            entity: Entity to add
            
        Returns:
            Node ID in the graph
        """
        # Convert entity to graph node
        graph_node = GraphNode.from_entity(entity)
        
        # Add to graph
        node_id, attributes = graph_node.to_networkx_node()
        self.graph.add_node(node_id, **attributes)
        
        return node_id
    
    def add_relationship(self, relationship: Relationship) -> Tuple[str, str]:
        """
        Add a relationship to the graph as an edge.
        
        Args:
            relationship: Relationship to add
            
        Returns:
            Tuple of (source_id, target_id) in the graph
        """
        # Convert relationship to graph edge
        graph_edge = GraphEdge.from_relationship(relationship)
        
        # Add to graph
        source_id, target_id, attributes = graph_edge.to_networkx_edge()
        
        # Check if the nodes exist
        if not self.graph.has_node(source_id):
            self.logger.warning(f"Source node {source_id} not found when adding relationship")
            # Create placeholder node
            self.graph.add_node(source_id, node_type="unknown", label="Unknown Entity")
            
        if not self.graph.has_node(target_id):
            self.logger.warning(f"Target node {target_id} not found when adding relationship")
            # Create placeholder node
            self.graph.add_node(target_id, node_type="unknown", label="Unknown Entity")
        
        # Add the edge
        self.graph.add_edge(source_id, target_id, **attributes)
        
        # Add reverse edge if bidirectional
        if self.bidirectional_relationships and relationship.bidirectional:
            # Create a copy of attributes for the reverse edge
            reverse_attributes = attributes.copy()
            reverse_attributes["is_reverse"] = True
            
            self.graph.add_edge(target_id, source_id, **reverse_attributes)
            
        return (source_id, target_id)
    
    def add_document_node(self, document: DocumentReference, 
                         entities: Optional[List[Entity]] = None) -> str:
        """
        Add a document node to the graph and connect it to entities.
        
        Args:
            document: Document reference to add
            entities: Optional list of entities to connect to the document node
            
        Returns:
            Document node ID in the graph
        """
        # Create document node
        doc_node_id = f"doc:{document.document_id}"
        
        # Add document node to graph
        self.graph.add_node(
            doc_node_id,
            node_type="document",
            label=document.title,
            document_id=document.document_id,
            document_type=document.document_type,
            source_path=document.source_path,
            publication_date=document.publication_date.isoformat() if document.publication_date else None,
            authors=document.authors
        )
        
        # Connect document to entities
        if entities:
            for entity in entities:
                if entity.source_document_id == document.document_id:
                    self.graph.add_edge(
                        doc_node_id,
                        entity.id,
                        edge_type="contains",
                        weight=1.0,
                        bidirectional=False
                    )
                    
        return doc_node_id
    
    def add_section_nodes(self, document: DocumentReference, 
                         sections: List[Dict[str, Any]],
                         entities: Optional[List[Entity]] = None) -> List[str]:
        """
        Add section nodes to the graph and connect them to entities and the document.
        
        Args:
            document: Document reference
            sections: List of section dictionaries with name, level, start, end
            entities: Optional list of entities to connect to section nodes
            
        Returns:
            List of section node IDs in the graph
        """
        if not self.include_section_nodes:
            return []
            
        section_node_ids = []
        doc_node_id = f"doc:{document.document_id}"
        
        # Add section nodes
        for i, section in enumerate(sections):
            section_id = f"sec:{document.document_id}:{i}"
            section_name = section.get("name", f"Section {i}")
            section_level = section.get("level", 1)
            
            # Add section node
            self.graph.add_node(
                section_id,
                node_type="section",
                label=section_name,
                document_id=document.document_id,
                section_level=section_level,
                section_index=i
            )
            section_node_ids.append(section_id)
            
            # Connect section to document
            if self.include_document_nodes and self.graph.has_node(doc_node_id):
                self.graph.add_edge(
                    doc_node_id,
                    section_id,
                    edge_type="has_section",
                    weight=1.0,
                    bidirectional=False
                )
                
            # Connect to parent section if applicable
            if i > 0 and section_level > sections[i-1].get("level", 1):
                parent_id = section_node_ids[-2]  # Previous section
                self.graph.add_edge(
                    parent_id,
                    section_id,
                    edge_type="has_subsection",
                    weight=1.0,
                    bidirectional=False
                )
                
            # Connect section to entities
            if entities:
                section_start = section.get("start", 0)
                section_end = section.get("end", 0)
                
                for entity in entities:
                    for position in entity.positions:
                        entity_start = position.get("start", 0)
                        entity_end = position.get("end", 0)
                        
                        # Check if entity is in this section
                        if (entity_start >= section_start and entity_start < section_end) or \
                           (entity_end > section_start and entity_end <= section_end):
                            self.graph.add_edge(
                                section_id,
                                entity.id,
                                edge_type="contains",
                                weight=1.0,
                                bidirectional=False
                            )
                            break  # Only add one edge per entity
                            
        return section_node_ids
    
    def merge_graphs(self, other_graph: nx.Graph) -> nx.Graph:
        """
        Merge another graph into this graph.
        
        Args:
            other_graph: NetworkX graph to merge
            
        Returns:
            The merged graph
        """
        self.logger.info(f"Merging graph with {other_graph.number_of_nodes()} nodes and {other_graph.number_of_edges()} edges")
        
        # Create a copy of the current graph
        merged_graph = self.graph.copy()
        
        # Add nodes from other graph
        for node, attrs in other_graph.nodes(data=True):
            if merged_graph.has_node(node):
                # Update attributes of existing node
                for key, value in attrs.items():
                    if key not in merged_graph.nodes[node]:
                        merged_graph.nodes[node][key] = value
                    elif key == "source_document_ids" and isinstance(value, list):
                        # Merge source document IDs
                        existing_ids = set(merged_graph.nodes[node].get("source_document_ids", []))
                        existing_ids.update(value)
                        merged_graph.nodes[node]["source_document_ids"] = list(existing_ids)
            else:
                # Add new node
                merged_graph.add_node(node, **attrs)
                
        # Add edges from other graph
        for u, v, attrs in other_graph.edges(data=True):
            if merged_graph.has_edge(u, v):
                # Update attributes of existing edge
                for key, value in attrs.items():
                    if key not in merged_graph[u][v]:
                        merged_graph[u][v][key] = value
                    elif key == "weight" and isinstance(value, (int, float)):
                        # Combine weights
                        merged_graph[u][v]["weight"] = (merged_graph[u][v]["weight"] + value) / 2
                    elif key == "source_document_ids" and isinstance(value, list):
                        # Merge source document IDs
                        existing_ids = set(merged_graph[u][v].get("source_document_ids", []))
                        existing_ids.update(value)
                        merged_graph[u][v]["source_document_ids"] = list(existing_ids)
            else:
                # Add new edge
                merged_graph.add_edge(u, v, **attrs)
                
        # Update graph metadata
        merged_graph.graph["merged_at"] = datetime.now().isoformat()
        merged_graph.graph["merged_graphs"] = merged_graph.graph.get("merged_graphs", 0) + 1
        
        self.logger.info(f"Merged graph has {merged_graph.number_of_nodes()} nodes and {merged_graph.number_of_edges()} edges")
        
        # Update the current graph
        self.graph = merged_graph
        
        return self.graph
    
    def _optimize_for_retrieval(self) -> None:
        """
        Optimize the graph structure for retrieval operations.
        
        This includes:
        - Adding index attributes for faster lookup
        - Precomputing centrality measures
        - Adding connection strength attributes
        """
        self.logger.info("Optimizing graph for retrieval")
        
        # Add node indices
        for i, node in enumerate(self.graph.nodes()):
            self.graph.nodes[node]["index"] = i
            
        # Compute degree centrality for all nodes
        try:
            centrality = nx.degree_centrality(self.graph)
            nx.set_node_attributes(self.graph, centrality, "centrality")
        except Exception as e:
            self.logger.warning(f"Failed to compute centrality: {str(e)}")
            
        # Add connection strength based on edge weights and node degrees
        for u, v, attrs in self.graph.edges(data=True):
            weight = attrs.get("weight", 1.0)
            # Get the degree values in a way that works for both integer and tuple returns
            u_degree_call = self.graph.degree(u)
            v_degree_call = self.graph.degree(v)
            
            # Handle both possible return types (int or tuple)
            if isinstance(u_degree_call, int):
                u_degree_val = u_degree_call
            elif isinstance(u_degree_call, tuple) and len(u_degree_call) == 2 and u_degree_call[0] == u:
                u_degree_val = u_degree_call[1]
            else:
                # Fallback, try to get it directly
                degree_u = self.graph.degree(u)
                u_degree_val = degree_u if isinstance(degree_u, int) else 1
                
            if isinstance(v_degree_call, int):
                v_degree_val = v_degree_call
            elif isinstance(v_degree_call, tuple) and len(v_degree_call) == 2 and v_degree_call[0] == v:
                v_degree_val = v_degree_call[1]
            else:
                # Fallback, try to get it directly
                degree_v = self.graph.degree(v)
                v_degree_val = degree_v if isinstance(degree_v, int) else 1
            
            # Now we have numeric values that can be safely added
            connection_strength = weight * (2.0 / (float(u_degree_val) + float(v_degree_val)))
            self.graph[u][v]["connection_strength"] = connection_strength
            
        self.logger.info("Graph optimization complete")
    
    def save_graph(self, filename: Optional[str] = None, document: Optional[DocumentReference] = None) -> str:
        """
        Save the graph to disk.
        
        Args:
            filename: Optional filename to save to (default: auto-generated)
            document: Optional document reference for caching
            
        Returns:
            Path to the saved graph file
        """
        if not filename:
            # Generate filename based on timestamp or document ID
            if document:
                filename = f"graph_{document.document_id}.pkl"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"graph_{timestamp}.pkl"
            
        # Ensure the filename has the correct extension
        if not filename.endswith(".pkl"):
            filename += ".pkl"
            
        # Create full path
        filepath = os.path.join(self.graph_dir, filename)
        
        # Save the graph
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.graph, f)
                
            # Also cache the graph if document is provided and caching is enabled
            if document and self.use_redis_cache and self.cache_manager:
                self.cache_manager.cache_document_graph(self.graph, document)
                self.logger.info(f"Cached graph for document {document.document_id}")
                
            self.logger.info(f"Saved graph to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to save graph: {str(e)}")
            raise
    
    def load_graph(self, filepath: str, document: Optional[DocumentReference] = None) -> nx.Graph:
        """
        Load a graph from disk or cache.
        
        Args:
            filepath: Path to the graph file
            document: Optional document reference for cache lookup
            
        Returns:
            The loaded NetworkX graph
        """
        # Try to load from cache first if document is provided and caching is enabled
        if document and self.use_redis_cache and self.cache_manager:
            cached_graph = self.cache_manager.get_document_graph(document)
            if cached_graph:
                self.logger.info(f"Loaded graph for document {document.document_id} from cache")
                self.graph = cached_graph
                return self.graph
        
        try:
            with open(filepath, 'rb') as f:
                self.graph = pickle.load(f)
                
            # Cache the loaded graph if document is provided and caching is enabled
            if document and self.use_redis_cache and self.cache_manager:
                self.cache_manager.cache_document_graph(self.graph, document)
                self.logger.info(f"Cached loaded graph for document {document.document_id}")
                
            self.logger.info(f"Loaded graph from {filepath} with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return self.graph
        except Exception as e:
            self.logger.error(f"Failed to load graph: {str(e)}")
            raise
    
    def save_graph_metadata(self, metadata_path: Optional[str] = None) -> str:
        """
        Save graph metadata to a JSON file.
        
        Args:
            metadata_path: Optional path for the metadata file
            
        Returns:
            Path to the saved metadata file
        """
        if not metadata_path:
            # Generate filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata_path = os.path.join(self.graph_dir, f"graph_{timestamp}_metadata.json")
            
        # Collect metadata
        metadata = {
            "graph_type": self.graph_type,
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "created_at": self.graph.graph.get("created_at", datetime.now().isoformat()),
            "node_types": {},
            "edge_types": {},
            "document_count": 0
        }
        
        # Count node types
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("node_type", "unknown")
            metadata["node_types"][node_type] = metadata["node_types"].get(node_type, 0) + 1
            
            # Count documents
            if node_type == "document":
                metadata["document_count"] += 1
                
        # Count edge types
        for u, v, attrs in self.graph.edges(data=True):
            edge_type = attrs.get("edge_type", "unknown")
            metadata["edge_types"][edge_type] = metadata["edge_types"].get(edge_type, 0) + 1
            
        # Save metadata
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"Saved graph metadata to {metadata_path}")
            return metadata_path
        except Exception as e:
            self.logger.error(f"Failed to save graph metadata: {str(e)}")
            raise
    
    def get_entity_subgraph(self, entity_id: str, depth: int = 1, document: Optional[DocumentReference] = None) -> nx.Graph:
        """
        Get a subgraph centered on a specific entity.
        
        Args:
            entity_id: ID of the entity to center the subgraph on
            depth: Number of hops to include in the subgraph
            document: Optional document reference for caching
            
        Returns:
            Subgraph centered on the specified entity
        """
        # Check cache first if document is provided and caching is enabled
        if document and self.use_redis_cache and self.cache_manager:
            # If we have an Entity object instead of just an ID, we can use it directly
            if isinstance(entity_id, Entity):
                entity = entity_id
                entity_id = entity.id
            else:
                # Create a minimal Entity object for caching purposes
                entity_obj = Entity(text=str(entity_id), entity_type="unknown", id=str(entity_id))
                entity = entity_obj  # type: ignore
                
            cached_subgraph = self.cache_manager.get_entity_subgraph(document, entity, depth)
            if cached_subgraph:
                self.logger.info(f"Retrieved entity subgraph for {entity_id} from cache")
                return cached_subgraph
        
        if not self.graph.has_node(entity_id):
            self.logger.warning(f"Entity {entity_id} not found in graph")
            # Create an empty graph of the same type as self.graph
            if isinstance(self.graph, nx.DiGraph):
                return nx.DiGraph()
            elif isinstance(self.graph, nx.MultiGraph):
                return nx.MultiGraph()
            elif isinstance(self.graph, nx.MultiDiGraph):
                return nx.MultiDiGraph()
            else:
                return nx.Graph()
        
        # Get all nodes within the specified depth
        nodes: Set[Any] = {entity_id}
        current_nodes: Set[Any] = {entity_id}
        
        for _ in range(depth):
            next_nodes: Set[Any] = set()
            for node in current_nodes:
                next_nodes.update(self.graph.neighbors(node))
            current_nodes = next_nodes - nodes  # Avoid revisiting nodes
            nodes.update(current_nodes)
            if not current_nodes:  # No more nodes to explore
                break
        
        # Create the subgraph
        subgraph: Union[nx.DiGraph, nx.Graph, nx.MultiGraph, nx.MultiDiGraph] = self.graph.subgraph(nodes).copy()  # type: ignore
        
        # Set the central entity attribute
        if entity_id in subgraph.nodes:
            subgraph.nodes[entity_id]["is_central"] = True
        
        # Cache the subgraph if document is provided and caching is enabled
        if document and self.use_redis_cache and self.cache_manager:
            if isinstance(entity_id, str):
                # Create a minimal Entity object for caching purposes
                entity_obj = Entity(text=str(entity_id), entity_type="unknown", id=str(entity_id))
                entity = entity_obj  # type: ignore
            self.cache_manager.cache_entity_subgraph(subgraph, document, entity, depth)
            self.logger.info(f"Cached entity subgraph for {entity_id}")
        
        return subgraph
        
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current graph.
        
        Returns:
            Dictionary of graph statistics
        """
        stats = {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "graph_type": self.graph_type,
            "is_directed": self.graph.is_directed(),
            "is_multigraph": self.graph.is_multigraph(),
            "node_types": {},
            "edge_types": {},
            "document_count": 0,
            "entity_count": 0,
            "relationship_count": 0
        }
        
        # Count node types
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("node_type", "unknown")
            stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
            
            # Count specific types
            if node_type == "document":
                stats["document_count"] += 1
            elif node_type != "section":
                stats["entity_count"] += 1
                
        # Count edge types
        for u, v, attrs in self.graph.edges(data=True):
            edge_type = attrs.get("edge_type", "unknown")
            stats["edge_types"][edge_type] = stats["edge_types"].get(edge_type, 0) + 1
            
            # Count relationships
            if edge_type not in ["contains", "has_section", "has_subsection"]:
                stats["relationship_count"] += 1
                
        # Calculate density
        try:
            stats["density"] = nx.density(self.graph)
        except Exception:
            stats["density"] = 0
            
        # Calculate average degree
        if stats["node_count"] > 0:
            # Fix incompatible argument type by handling each node degree separately
            degrees = []
            degree_view = self.graph.degree()
            
            # Handle different graph types that might return different degree objects
            if hasattr(degree_view, '__iter__'):
                for node_degree_pair in degree_view:
                    # Each item should be a tuple of (node, degree)
                    if isinstance(node_degree_pair, tuple) and len(node_degree_pair) == 2:
                        node, degree = node_degree_pair
                        degrees.append(degree)
            else:
                # For graph types where degree() returns a dict-like view
                for node in self.graph.nodes():
                    degrees.append(self.graph.degree(node))
                    
            total_degree = sum(degrees) if degrees else 0
            stats["average_degree"] = total_degree / stats["node_count"]
        else:
            stats["average_degree"] = 0
        
        # Add cache statistics if caching is enabled
        if self.use_redis_cache and self.cache_manager:
            stats["cache"] = self.get_cache_stats()
            
        return stats
    
    def invalidate_cache(self, document: Optional[DocumentReference] = None) -> int:
        """
        Invalidate cached graphs for a document or all cached graphs.
        
        Args:
            document: Optional document to invalidate caches for. If None, invalidates all caches.
            
        Returns:
            Number of invalidated cache entries
        """
        if not self.use_redis_cache or not self.cache_manager:
            return 0
            
        if document:
            count = self.cache_manager.invalidate_document_caches(document.document_id)
            self.logger.info(f"Invalidated {count} cache entries for document {document.document_id}")
        else:
            count = self.cache_manager.clear_all_caches()
            self.logger.info(f"Invalidated all {count} cache entries")
            
        return count
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics or empty dict if caching is disabled
        """
        if not self.use_redis_cache or not self.cache_manager:
            return {}
            
        return self.cache_manager.get_cache_stats()
