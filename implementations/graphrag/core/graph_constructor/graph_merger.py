"""
Graph Merger for GraphRAG

This module implements the GraphMerger class that combines multiple document graphs
into a unified knowledge graph in the GraphRAG implementation.
"""

import networkx as nx
from typing import Dict, List, Optional, Any, Set, Tuple, Union
import logging
import os
from datetime import datetime
import pickle
import json

from ..models.entity import Entity
from ..models.relationship import Relationship
from ..models.document import DocumentReference
from .graph_constructor import GraphConstructor
from .document_graph import DocumentGraph


class GraphMerger:
    """
    Class for merging multiple document graphs into a unified knowledge graph.
    
    The GraphMerger combines document graphs, resolves entity conflicts,
    and creates a comprehensive knowledge graph that represents the entire
    document collection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the graph merger.
        
        Args:
            config: Configuration dictionary with the following options:
                - graph_dir: Directory for storing graph data
                - merge_strategy: Strategy for merging entities ("simple", "weighted", "semantic")
                - entity_resolution_threshold: Threshold for entity resolution
                - include_document_nodes: Whether to include document nodes
                - optimize_merged_graph: Whether to optimize the merged graph
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Get configuration options
        self.merge_strategy = self.config.get("merge_strategy", "weighted")
        self.entity_resolution_threshold = self.config.get("entity_resolution_threshold", 0.8)
        self.optimize_merged_graph = self.config.get("optimize_merged_graph", True)
        
        # Initialize graph constructor
        self.graph_constructor = GraphConstructor(self.config)
        
        # Set up graph directory
        self.graph_dir = self.config.get(
            "graph_dir",
            os.path.join("/home/todd/ML-Lab/Olympus/limnos/data/implementations/graphrag/graphs")
        )
        self.merged_graph_dir = os.path.join(self.graph_dir, "merged_graphs")
        os.makedirs(self.merged_graph_dir, exist_ok=True)
        
        # Initialize the merged graph
        self.merged_graph = None
        self.document_graphs = {}
        self.entity_mapping = {}  # Maps entity IDs to canonical entity IDs
        
        self.logger.info(f"Initialized GraphMerger with merge strategy: {self.merge_strategy}")
    
    def add_document_graph(self, document_graph: DocumentGraph) -> None:
        """
        Add a document graph to the merger.
        
        Args:
            document_graph: DocumentGraph instance to add
        """
        doc_id = document_graph.document.document_id
        self.document_graphs[doc_id] = document_graph
        self.logger.info(f"Added document graph for {doc_id} to merger")
    
    def merge_graphs(self, document_graphs: Optional[List[DocumentGraph]] = None) -> nx.Graph:
        """
        Merge document graphs into a unified knowledge graph.
        
        Args:
            document_graphs: Optional list of document graphs to merge
                (if None, uses previously added document graphs)
            
        Returns:
            The merged knowledge graph
        """
        # Use provided document graphs or previously added ones
        if document_graphs:
            for graph in document_graphs:
                self.add_document_graph(graph)
                
        if not self.document_graphs:
            self.logger.error("No document graphs to merge")
            raise ValueError("No document graphs to merge")
            
        self.logger.info(f"Merging {len(self.document_graphs)} document graphs")
        
        # Initialize a new graph
        self.merged_graph = nx.Graph() if self.graph_constructor.graph_type == "undirected" else nx.DiGraph()
        self.merged_graph.graph["created_at"] = datetime.now().isoformat()
        self.merged_graph.graph["graph_type"] = self.graph_constructor.graph_type
        self.merged_graph.graph["framework"] = "graphrag"
        self.merged_graph.graph["document_count"] = len(self.document_graphs)
        
        # Reset entity mapping
        self.entity_mapping = {}
        
        # First pass: add all document nodes
        if self.graph_constructor.include_document_nodes:
            self._add_document_nodes()
            
        # Second pass: resolve entities and add to merged graph
        self._resolve_and_add_entities()
        
        # Third pass: add relationships with resolved entity IDs
        self._add_relationships()
        
        # Optimize the merged graph if configured
        if self.optimize_merged_graph:
            self._optimize_merged_graph()
            
        self.logger.info(f"Created merged graph with {self.merged_graph.number_of_nodes()} nodes "
                        f"and {self.merged_graph.number_of_edges()} edges")
        
        return self.merged_graph
    
    def _add_document_nodes(self) -> None:
        """
        Add document nodes from all document graphs to the merged graph.
        """
        for doc_id, doc_graph in self.document_graphs.items():
            if doc_graph.graph is None:
                self.logger.warning(f"Document graph for {doc_id} is not initialized, skipping")
                continue
                
            # Find document node in the document graph
            doc_node_id = f"doc:{doc_id}"
            if doc_graph.graph.has_node(doc_node_id):
                # Copy document node attributes
                doc_attrs = doc_graph.graph.nodes[doc_node_id]
                self.merged_graph.add_node(doc_node_id, **doc_attrs)
                self.logger.debug(f"Added document node for {doc_id} to merged graph")
    
    def _resolve_and_add_entities(self) -> None:
        """
        Resolve entity conflicts and add entities to the merged graph.
        """
        # First, collect all entities from all document graphs
        all_entities = {}  # entity_id -> (entity, doc_id, attributes)
        
        for doc_id, doc_graph in self.document_graphs.items():
            if doc_graph.graph is None:
                continue
                
            # Get all entity nodes (excluding document and section nodes)
            for node, attrs in doc_graph.graph.nodes(data=True):
                if attrs.get("node_type") not in ["document", "section"]:
                    all_entities[node] = (node, doc_id, attrs)
        
        # Resolve entities based on the merge strategy
        if self.merge_strategy == "simple":
            self._simple_entity_resolution(all_entities)
        elif self.merge_strategy == "weighted":
            self._weighted_entity_resolution(all_entities)
        elif self.merge_strategy == "semantic":
            self._semantic_entity_resolution(all_entities)
        else:
            self.logger.warning(f"Unknown merge strategy: {self.merge_strategy}, using simple")
            self._simple_entity_resolution(all_entities)
    
    def _simple_entity_resolution(self, all_entities: Dict[str, Tuple[str, str, Dict[str, Any]]]) -> None:
        """
        Simple entity resolution based on exact text matches.
        
        Args:
            all_entities: Dictionary mapping entity IDs to (entity_id, doc_id, attributes)
        """
        # Group entities by text
        entities_by_text = {}
        
        for entity_id, (_, doc_id, attrs) in all_entities.items():
            entity_text = attrs.get("label", "").lower()
            entity_type = attrs.get("entity_type", "unknown")
            
            if not entity_text:
                # No text to match, keep as separate entity
                self.entity_mapping[entity_id] = entity_id
                self.merged_graph.add_node(entity_id, **attrs)
                continue
                
            # Create a key from text and type
            key = (entity_text, entity_type)
            
            if key not in entities_by_text:
                entities_by_text[key] = []
                
            entities_by_text[key].append((entity_id, doc_id, attrs))
        
        # For each group, select a canonical entity and map others to it
        for key, entities in entities_by_text.items():
            if not entities:
                continue
                
            # Use the first entity as canonical
            canonical_id, _, canonical_attrs = entities[0]
            
            # Add canonical entity to merged graph
            merged_attrs = canonical_attrs.copy()
            merged_attrs["source_document_ids"] = [entities[0][1]]  # Start with first doc_id
            self.merged_graph.add_node(canonical_id, **merged_attrs)
            
            # Map all entities to the canonical one
            self.entity_mapping[canonical_id] = canonical_id  # Map to itself
            
            # Process other entities in the group
            for other_id, other_doc_id, _ in entities[1:]:
                self.entity_mapping[other_id] = canonical_id
                
                # Add document ID to the canonical entity
                if "source_document_ids" in self.merged_graph.nodes[canonical_id]:
                    self.merged_graph.nodes[canonical_id]["source_document_ids"].append(other_doc_id)
                else:
                    self.merged_graph.nodes[canonical_id]["source_document_ids"] = [other_doc_id]
    
    def _weighted_entity_resolution(self, all_entities: Dict[str, Tuple[str, str, Dict[str, Any]]]) -> None:
        """
        Weighted entity resolution based on text similarity and entity type.
        
        Args:
            all_entities: Dictionary mapping entity IDs to (entity_id, doc_id, attributes)
        """
        # Group entities by type first
        entities_by_type = {}
        
        for entity_id, (_, doc_id, attrs) in all_entities.items():
            entity_type = attrs.get("entity_type", "unknown")
            
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
                
            entities_by_type[entity_type].append((entity_id, doc_id, attrs))
        
        # Process each type group separately
        for entity_type, entities in entities_by_type.items():
            # Further group by text similarity
            processed = set()
            
            for i, (entity_id, doc_id, attrs) in enumerate(entities):
                if entity_id in processed:
                    continue
                    
                entity_text = attrs.get("label", "").lower()
                if not entity_text:
                    # No text to match, keep as separate entity
                    self.entity_mapping[entity_id] = entity_id
                    self.merged_graph.add_node(entity_id, **attrs)
                    processed.add(entity_id)
                    continue
                
                # Find similar entities
                similar_entities = [(entity_id, doc_id, attrs)]
                processed.add(entity_id)
                
                for j, (other_id, other_doc_id, other_attrs) in enumerate(entities):
                    if other_id in processed or i == j:
                        continue
                        
                    other_text = other_attrs.get("label", "").lower()
                    
                    # Calculate text similarity
                    similarity = self._calculate_text_similarity(entity_text, other_text)
                    
                    if similarity >= self.entity_resolution_threshold:
                        similar_entities.append((other_id, other_doc_id, other_attrs))
                        processed.add(other_id)
                
                # Select the canonical entity (highest confidence or first)
                canonical_id, canonical_doc_id, canonical_attrs = max(
                    similar_entities, 
                    key=lambda x: x[2].get("confidence", 0.0)
                )
                
                # Add canonical entity to merged graph
                merged_attrs = canonical_attrs.copy()
                merged_attrs["source_document_ids"] = [d for _, d, _ in similar_entities]
                
                # Calculate average confidence if available
                confidences = [a.get("confidence", 0.0) for _, _, a in similar_entities]
                if confidences:
                    merged_attrs["confidence"] = sum(confidences) / len(confidences)
                    
                self.merged_graph.add_node(canonical_id, **merged_attrs)
                
                # Map all entities to the canonical one
                for other_id, _, _ in similar_entities:
                    self.entity_mapping[other_id] = canonical_id
    
    def _semantic_entity_resolution(self, all_entities: Dict[str, Tuple[str, str, Dict[str, Any]]]) -> None:
        """
        Semantic entity resolution using embeddings (if available).
        
        Args:
            all_entities: Dictionary mapping entity IDs to (entity_id, doc_id, attributes)
        """
        # Check if embeddings are available
        has_embeddings = any("embedding" in attrs for _, _, attrs in all_entities.values())
        
        if not has_embeddings:
            self.logger.warning("No embeddings found for semantic entity resolution, falling back to weighted")
            return self._weighted_entity_resolution(all_entities)
            
        # Group entities by type first
        entities_by_type = {}
        
        for entity_id, (_, doc_id, attrs) in all_entities.items():
            entity_type = attrs.get("entity_type", "unknown")
            
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
                
            entities_by_type[entity_type].append((entity_id, doc_id, attrs))
        
        # Process each type group separately
        for entity_type, entities in entities_by_type.items():
            # Further group by semantic similarity
            processed = set()
            
            for i, (entity_id, doc_id, attrs) in enumerate(entities):
                if entity_id in processed:
                    continue
                    
                entity_embedding = attrs.get("embedding")
                if entity_embedding is None:
                    # No embedding, use weighted resolution for this entity
                    if entity_id not in self.entity_mapping:
                        similar_entities = [(entity_id, doc_id, attrs)]
                        processed.add(entity_id)
                        
                        for j, (other_id, other_doc_id, other_attrs) in enumerate(entities):
                            if other_id in processed or i == j:
                                continue
                                
                            entity_text = attrs.get("label", "").lower()
                            other_text = other_attrs.get("label", "").lower()
                            
                            # Calculate text similarity
                            similarity = self._calculate_text_similarity(entity_text, other_text)
                            
                            if similarity >= self.entity_resolution_threshold:
                                similar_entities.append((other_id, other_doc_id, other_attrs))
                                processed.add(other_id)
                        
                        # Select canonical entity
                        canonical_id, _, canonical_attrs = max(
                            similar_entities, 
                            key=lambda x: x[2].get("confidence", 0.0)
                        )
                        
                        # Add to merged graph
                        merged_attrs = canonical_attrs.copy()
                        merged_attrs["source_document_ids"] = [d for _, d, _ in similar_entities]
                        self.merged_graph.add_node(canonical_id, **merged_attrs)
                        
                        # Map entities
                        for other_id, _, _ in similar_entities:
                            self.entity_mapping[other_id] = canonical_id
                    continue
                
                # Find semantically similar entities
                similar_entities = [(entity_id, doc_id, attrs)]
                processed.add(entity_id)
                
                for j, (other_id, other_doc_id, other_attrs) in enumerate(entities):
                    if other_id in processed or i == j:
                        continue
                        
                    other_embedding = other_attrs.get("embedding")
                    if other_embedding is None:
                        continue
                        
                    # Calculate embedding similarity
                    similarity = self._calculate_embedding_similarity(entity_embedding, other_embedding)
                    
                    if similarity >= self.entity_resolution_threshold:
                        similar_entities.append((other_id, other_doc_id, other_attrs))
                        processed.add(other_id)
                
                # Select the canonical entity (highest confidence or first)
                canonical_id, canonical_doc_id, canonical_attrs = max(
                    similar_entities, 
                    key=lambda x: x[2].get("confidence", 0.0)
                )
                
                # Add canonical entity to merged graph
                merged_attrs = canonical_attrs.copy()
                merged_attrs["source_document_ids"] = [d for _, d, _ in similar_entities]
                
                # Calculate average confidence if available
                confidences = [a.get("confidence", 0.0) for _, _, a in similar_entities]
                if confidences:
                    merged_attrs["confidence"] = sum(confidences) / len(confidences)
                    
                self.merged_graph.add_node(canonical_id, **merged_attrs)
                
                # Map all entities to the canonical one
                for other_id, _, _ in similar_entities:
                    self.entity_mapping[other_id] = canonical_id
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple Jaccard similarity for tokens
        if not text1 or not text2:
            return 0.0
            
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def _calculate_embedding_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity between 0 and 1
        """
        import numpy as np
        
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _add_relationships(self) -> None:
        """
        Add relationships to the merged graph, using resolved entity IDs.
        """
        # Collect all relationships from document graphs
        for doc_id, doc_graph in self.document_graphs.items():
            if doc_graph.graph is None:
                continue
                
            # Get all edges that are not document-entity connections
            for u, v, attrs in doc_graph.graph.edges(data=True):
                edge_type = attrs.get("edge_type", "unknown")
                
                # Skip document-entity and section-entity connections
                if edge_type in ["contains", "has_section", "has_subsection"]:
                    continue
                    
                # Get resolved entity IDs
                resolved_u = self.entity_mapping.get(u, u)
                resolved_v = self.entity_mapping.get(v, v)
                
                # Skip self-loops created by entity resolution
                if resolved_u == resolved_v:
                    continue
                    
                # Check if both nodes exist in the merged graph
                if not self.merged_graph.has_node(resolved_u) or not self.merged_graph.has_node(resolved_v):
                    continue
                    
                # Check if the edge already exists
                if self.merged_graph.has_edge(resolved_u, resolved_v):
                    # Update existing edge
                    edge_attrs = self.merged_graph[resolved_u][resolved_v]
                    
                    # Combine weights
                    if "weight" in attrs and "weight" in edge_attrs:
                        edge_attrs["weight"] = (edge_attrs["weight"] + attrs["weight"]) / 2
                        
                    # Update source document IDs
                    if "source_document_ids" in edge_attrs:
                        if doc_id not in edge_attrs["source_document_ids"]:
                            edge_attrs["source_document_ids"].append(doc_id)
                    else:
                        edge_attrs["source_document_ids"] = [doc_id]
                else:
                    # Add new edge
                    edge_attrs = attrs.copy()
                    edge_attrs["source_document_ids"] = [doc_id]
                    self.merged_graph.add_edge(resolved_u, resolved_v, **edge_attrs)
                    
                    # Add bidirectional edge if needed
                    if attrs.get("bidirectional", False) and not self.merged_graph.has_edge(resolved_v, resolved_u):
                        reverse_attrs = attrs.copy()
                        reverse_attrs["is_reverse"] = True
                        reverse_attrs["source_document_ids"] = [doc_id]
                        self.merged_graph.add_edge(resolved_v, resolved_u, **reverse_attrs)
    
    def _optimize_merged_graph(self) -> None:
        """
        Optimize the merged graph for retrieval operations.
        """
        self.logger.info("Optimizing merged graph for retrieval")
        
        # Add node indices
        for i, node in enumerate(self.merged_graph.nodes()):
            self.merged_graph.nodes[node]["index"] = i
            
        # Compute centrality measures
        try:
            # Degree centrality
            centrality = nx.degree_centrality(self.merged_graph)
            nx.set_node_attributes(self.merged_graph, centrality, "centrality")
            
            # Betweenness centrality (for important nodes)
            betweenness = nx.betweenness_centrality(self.merged_graph, k=min(100, len(self.merged_graph)))
            nx.set_node_attributes(self.merged_graph, betweenness, "betweenness_centrality")
        except Exception as e:
            self.logger.warning(f"Failed to compute centrality measures: {str(e)}")
            
        # Add connection strength based on edge weights and node degrees
        for u, v, attrs in self.merged_graph.edges(data=True):
            weight = attrs.get("weight", 1.0)
            u_degree = self.merged_graph.degree(u)
            v_degree = self.merged_graph.degree(v)
            
            # Calculate connection strength (higher for nodes with fewer connections)
            connection_strength = weight * (2.0 / (u_degree + v_degree))
            self.merged_graph[u][v]["connection_strength"] = connection_strength
            
        self.logger.info("Merged graph optimization complete")
    
    def save_merged_graph(self, filename: Optional[str] = None) -> str:
        """
        Save the merged graph to disk.
        
        Args:
            filename: Optional filename to save to (default: auto-generated)
            
        Returns:
            Path to the saved graph file
        """
        if self.merged_graph is None:
            self.logger.error("Cannot save merged graph: graph not initialized")
            raise ValueError("Merged graph not initialized")
            
        if not filename:
            # Generate filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"merged_graph_{timestamp}.pkl"
            
        # Ensure the filename has the correct extension
        if not filename.endswith(".pkl"):
            filename += ".pkl"
            
        # Create full path
        filepath = os.path.join(self.merged_graph_dir, filename)
        
        # Save the graph
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.merged_graph, f)
            self.logger.info(f"Saved merged graph to {filepath}")
            
            # Save metadata
            metadata_path = os.path.join(self.merged_graph_dir, f"{os.path.splitext(filename)[0]}_metadata.json")
            self._save_metadata(metadata_path)
            
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to save merged graph: {str(e)}")
            raise
    
    def load_merged_graph(self, filepath: str) -> nx.Graph:
        """
        Load a merged graph from disk.
        
        Args:
            filepath: Path to the graph file
            
        Returns:
            The loaded merged graph
        """
        try:
            with open(filepath, 'rb') as f:
                self.merged_graph = pickle.load(f)
            self.logger.info(f"Loaded merged graph from {filepath} with {self.merged_graph.number_of_nodes()} nodes and {self.merged_graph.number_of_edges()} edges")
            return self.merged_graph
        except Exception as e:
            self.logger.error(f"Failed to load merged graph: {str(e)}")
            raise
    
    def _save_metadata(self, metadata_path: str) -> None:
        """
        Save merged graph metadata to a JSON file.
        
        Args:
            metadata_path: Path for the metadata file
        """
        if self.merged_graph is None:
            return
            
        # Collect metadata
        metadata = {
            "node_count": self.merged_graph.number_of_nodes(),
            "edge_count": self.merged_graph.number_of_edges(),
            "document_count": len(self.document_graphs),
            "created_at": self.merged_graph.graph.get("created_at", datetime.now().isoformat()),
            "node_types": {},
            "edge_types": {},
            "document_ids": list(self.document_graphs.keys())
        }
        
        # Count node types
        for node, attrs in self.merged_graph.nodes(data=True):
            node_type = attrs.get("node_type", "unknown")
            metadata["node_types"][node_type] = metadata["node_types"].get(node_type, 0) + 1
            
        # Count edge types
        for u, v, attrs in self.merged_graph.edges(data=True):
            edge_type = attrs.get("edge_type", "unknown")
            metadata["edge_types"][edge_type] = metadata["edge_types"].get(edge_type, 0) + 1
            
        # Save metadata
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"Saved merged graph metadata to {metadata_path}")
        except Exception as e:
            self.logger.error(f"Failed to save merged graph metadata: {str(e)}")
    
    def get_merged_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the merged graph.
        
        Returns:
            Dictionary of graph statistics
        """
        if self.merged_graph is None:
            return {"status": "not_initialized"}
            
        # Get basic statistics
        stats = {
            "status": "initialized",
            "node_count": self.merged_graph.number_of_nodes(),
            "edge_count": self.merged_graph.number_of_edges(),
            "document_count": len(self.document_graphs),
            "is_directed": self.merged_graph.is_directed(),
            "node_types": {},
            "edge_types": {},
            "entity_count": 0,
            "relationship_count": 0
        }
        
        # Count node types
        for node, attrs in self.merged_graph.nodes(data=True):
            node_type = attrs.get("node_type", "unknown")
            stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
            
            # Count entities
            if node_type not in ["document", "section"]:
                stats["entity_count"] += 1
                
        # Count edge types
        for u, v, attrs in self.merged_graph.edges(data=True):
            edge_type = attrs.get("edge_type", "unknown")
            stats["edge_types"][edge_type] = stats["edge_types"].get(edge_type, 0) + 1
            
            # Count relationships
            if edge_type not in ["contains", "has_section", "has_subsection"]:
                stats["relationship_count"] += 1
                
        # Calculate density
        try:
            stats["density"] = nx.density(self.merged_graph)
        except Exception:
            stats["density"] = 0
            
        # Calculate average degree
        if stats["node_count"] > 0:
            total_degree = sum(dict(self.merged_graph.degree()).values())
            stats["average_degree"] = total_degree / stats["node_count"]
        else:
            stats["average_degree"] = 0
            
        return stats
