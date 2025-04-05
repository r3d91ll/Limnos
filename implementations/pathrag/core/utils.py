"""
PathRAG Utilities

This module provides utility functions for path extraction and processing in PathRAG.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import networkx as nx
from pathlib import Path

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean text for entity extraction.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might interfere with processing
    text = re.sub(r'[^\w\s.,;:!?\'"-]', ' ', text)
    
    return text.strip()

def merge_entity_lists(entity_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merge multiple lists of entities, avoiding duplicates.
    
    Args:
        entity_lists: List of entity lists to merge
        
    Returns:
        Merged list of entities
    """
    merged = {}
    
    for entities in entity_lists:
        for entity in entities:
            # Use entity text and type as key to avoid duplicates
            key = (entity["text"], entity["type"])
            
            if key not in merged:
                merged[key] = entity
    
    return list(merged.values())

def convert_graph_to_nx(entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> nx.MultiDiGraph:
    """
    Convert entities and relationships to a NetworkX graph.
    
    Args:
        entities: List of entity dictionaries
        relationships: List of relationship dictionaries
        
    Returns:
        NetworkX MultiDiGraph
    """
    G = nx.MultiDiGraph()
    
    # Add entities as nodes
    for entity in entities:
        G.add_node(
            entity["id"],
            text=entity["text"],
            type=entity["type"],
            metadata=entity
        )
    
    # Add relationships as edges
    for rel in relationships:
        source_id = rel["source"]["id"]
        target_id = rel["target"]["id"]
        
        # Skip if source or target is not in the graph
        if source_id not in G or target_id not in G:
            continue
            
        G.add_edge(
            source_id,
            target_id,
            id=rel["id"],
            type=rel["type"],
            weight=1.0 - rel.get("confidence", 0.5),  # Convert confidence to weight
            metadata=rel
        )
    
    return G

def visualize_paths(paths: List[Dict[str, Any]], output_file: Optional[str] = None) -> Optional[str]:
    """
    Visualize paths using NetworkX and matplotlib.
    
    Args:
        paths: List of path dictionaries
        output_file: Optional file path to save visualization
        
    Returns:
        Path to saved visualization file if output_file is provided
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create combined graph of all paths
        G = nx.DiGraph()
        
        for path in paths:
            for i in range(len(path["nodes"]) - 1):
                source = path["nodes"][i]
                target = path["nodes"][i + 1]
                
                # Extract text for nodes
                source_text = next((n["text"] for n in path["node_data"] if n["id"] == source), source)
                target_text = next((n["text"] for n in path["node_data"] if n["id"] == target), target)
                
                # Get edge info
                edge_info = next((e for e in path["edges"] if e["source"] == source and e["target"] == target), {})
                edge_type = edge_info.get("type", "RELATED_TO")
                
                # Add nodes and edge
                G.add_node(source, label=source_text)
                G.add_node(target, label=target_text)
                G.add_edge(source, target, label=edge_type)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7, arrows=True)
        
        # Draw labels
        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.axis('off')
        plt.title('PathRAG Paths Visualization')
        
        # Save or show
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            plt.close()
            return output_file
        else:
            plt.show()
            return None
            
    except ImportError:
        logger.warning("Matplotlib required for visualization")
        return None

def load_json_file(filepath: str) -> Any:
    """
    Load JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded JSON content
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data: Any, filepath: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save to
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
