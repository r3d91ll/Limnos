"""
Graph Serializer

Handles serialization and deserialization of NetworkX graphs for storage in Redis.
"""

import json
import pickle
import logging
import zlib
from typing import Dict, Any, Union, Optional
import networkx as nx

logger = logging.getLogger(__name__)

class GraphSerializer:
    """
    Serializes and deserializes NetworkX graphs for Redis storage.
    
    Provides multiple serialization formats with compression for efficient
    storage and retrieval of graph data.
    """
    
    FORMATS = ['pickle', 'json', 'adjacency']
    
    def __init__(self, format_type: str = 'pickle', compression_level: int = 6):
        """
        Initialize the serializer.
        
        Args:
            format_type: Serialization format ('pickle', 'json', or 'adjacency')
            compression_level: zlib compression level (0-9, 0=none, 9=max)
        """
        if format_type not in self.FORMATS:
            raise ValueError(f"Unknown format type: {format_type}. Use one of {self.FORMATS}")
        
        self.format_type = format_type
        self.compression_level = compression_level
        logger.info(f"Initialized GraphSerializer with {format_type} format, compression level {compression_level}")
    
    def serialize(self, graph: nx.Graph) -> bytes:
        """
        Serialize a NetworkX graph to binary format.
        
        Args:
            graph: The NetworkX graph to serialize
            
        Returns:
            bytes: The serialized graph data
        """
        try:
            # Serialize based on selected format
            if self.format_type == 'pickle':
                data = self._pickle_serialize(graph)
            elif self.format_type == 'json':
                data = self._json_serialize(graph)
            elif self.format_type == 'adjacency':
                data = self._adjacency_serialize(graph)
            else:
                raise ValueError(f"Unknown format type: {self.format_type}")
            
            # Add format info to header (first 2 bytes indicate format)
            format_indicator = {'pickle': b'PK', 'json': b'JS', 'adjacency': b'AD'}[self.format_type]
            
            # Compress if needed
            if self.compression_level > 0:
                compressed_data = zlib.compress(data, self.compression_level)
                # Add compression flag (3rd byte, 'C' = compressed)
                header = format_indicator + b'C'
                return header + compressed_data
            else:
                # Add no-compression flag (3rd byte, 'N' = not compressed)
                header = format_indicator + b'N'
                return header + data
        except Exception as e:
            logger.error(f"Error serializing graph: {e}")
            raise
    
    def deserialize(self, data: bytes) -> nx.Graph:
        """
        Deserialize binary data back to a NetworkX graph.
        
        Args:
            data: The serialized graph data
            
        Returns:
            nx.Graph: The deserialized NetworkX graph
        """
        try:
            # Extract format and compression info from header
            format_indicator = data[:2].decode('utf-8')
            compression_flag = data[2:3].decode('utf-8')
            
            # Map format indicator to format type
            format_map = {'PK': 'pickle', 'JS': 'json', 'AD': 'adjacency'}
            format_type = format_map.get(format_indicator)
            
            if not format_type:
                raise ValueError(f"Unknown format indicator: {format_indicator}")
            
            # Extract actual data
            data_payload = data[3:]
            
            # Decompress if needed
            if compression_flag == 'C':
                data_payload = zlib.decompress(data_payload)
            
            # Deserialize based on format
            if format_type == 'pickle':
                return self._pickle_deserialize(data_payload)
            elif format_type == 'json':
                return self._json_deserialize(data_payload)
            elif format_type == 'adjacency':
                return self._adjacency_deserialize(data_payload)
            else:
                raise ValueError(f"Unknown format type: {format_type}")
        except Exception as e:
            logger.error(f"Error deserializing graph: {e}")
            raise
    
    def _pickle_serialize(self, graph: nx.Graph) -> bytes:
        """Serialize using pickle protocol"""
        return pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _pickle_deserialize(self, data: bytes) -> nx.Graph:
        """Deserialize from pickle data"""
        return pickle.loads(data)
    
    def _json_serialize(self, graph: nx.Graph) -> bytes:
        """Serialize using JSON representation"""
        # Convert to node-link format
        data = nx.node_link_data(graph)
        json_str = json.dumps(data)
        return json_str.encode('utf-8')
    
    def _json_deserialize(self, data: bytes) -> nx.Graph:
        """Deserialize from JSON data"""
        json_str = data.decode('utf-8')
        data = json.loads(json_str)
        return nx.node_link_graph(data)
    
    def _adjacency_serialize(self, graph: nx.Graph) -> bytes:
        """Serialize using adjacency format"""
        # Get adjacency data
        adj_data = dict(nx.to_dict_of_dicts(graph))
        
        # Get node attributes
        node_attrs = {node: attrs for node, attrs in graph.nodes(data=True)}
        
        # Get graph attributes
        graph_attrs = dict(graph.graph)
        
        # Create a complete representation
        full_data = {
            'adjacency': adj_data,
            'node_attrs': node_attrs,
            'graph_attrs': graph_attrs,
            'directed': graph.is_directed(),
            'multigraph': graph.is_multigraph()
        }
        
        json_str = json.dumps(full_data)
        return json_str.encode('utf-8')
    
    def _adjacency_deserialize(self, data: bytes) -> nx.Graph:
        """Deserialize from adjacency format"""
        json_str = data.decode('utf-8')
        full_data = json.loads(json_str)
        
        # Create appropriate graph type
        directed = full_data.get('directed', False)
        multigraph = full_data.get('multigraph', False)
        
        if directed and multigraph:
            G = nx.MultiDiGraph()
        elif directed:
            G = nx.DiGraph()
        elif multigraph:
            G = nx.MultiGraph()
        else:
            G = nx.Graph()
        
        # Set graph attributes
        G.graph.update(full_data.get('graph_attrs', {}))
        
        # Add nodes with attributes
        for node, attrs in full_data.get('node_attrs', {}).items():
            G.add_node(node, **attrs)
        
        # Add edges with attributes
        adj_data = full_data.get('adjacency', {})
        for u, nbrs in adj_data.items():
            for v, data in nbrs.items():
                G.add_edge(u, v, **data)
        
        return G
