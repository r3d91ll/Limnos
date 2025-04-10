"""
Path Storage Manager for PathRAG

This module handles the storage and retrieval of path data in accordance with
the Limnos metadata architecture principles, maintaining clear separation between
universal and framework-specific metadata.
"""

import os
import json
import logging
import shutil
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional, Union, cast
from pathlib import Path as FilePath
import pickle
import time

from .path_structures import Path as RagPath, PathNode, PathEdge, PathIndex
from .path_vector_store import PathVectorStore

# Configure logging
logger = logging.getLogger(__name__)

class PathStorageManager:
    """
    Manages the storage and retrieval of path data in accordance with
    the Limnos metadata architecture principles.
    
    Ensures proper separation between universal and framework-specific metadata
    and maintains storage isolation for the PathRAG implementation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the path storage manager with configuration.
        
        Args:
            config: Configuration dictionary with options for the storage manager
        """
        self.config = config or {}
        
        # Base directory for universal metadata (source documents)
        self.source_documents_dir = self.config.get(
            'source_documents_dir', 
            os.path.join('/home/todd/ML-Lab/Olympus/limnos/data/source_documents')
        )
        
        # Base directory for PathRAG-specific data
        self.pathrag_data_dir = self.config.get(
            'pathrag_data_dir',
            os.path.join('/home/todd/ML-Lab/Olympus/limnos/data/implementations/pathrag')
        )
        
        # Directory for path data within the PathRAG directory
        self.paths_dir = os.path.join(self.pathrag_data_dir, 'paths')
        
        # Directory for path indexes
        self.indexes_dir = os.path.join(self.pathrag_data_dir, 'indexes')
        
        # Directory for path embeddings
        self.embeddings_dir = os.path.join(self.pathrag_data_dir, 'embeddings')
        
        # Create directories if they don't exist
        for directory in [self.paths_dir, self.indexes_dir, self.embeddings_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Path index for efficient retrieval
        self.path_index = PathIndex({
            'storage_dir': self.indexes_dir
        })
        
        # Vector store for semantic search
        self.vector_store = PathVectorStore({
            'storage_dir': self.embeddings_dir,
            'model_backend': self.config.get('model_backend', 'sentence_transformers'),
            'model_name': self.config.get('model_name', 'all-MiniLM-L6-v2'),
            'dimension': self.config.get('dimension', 768),
            'normalize_embeddings': self.config.get('normalize_embeddings', True)
        })
    
    def save_path(self, path: RagPath, embed: bool = True) -> str:
        """
        Save a path to the PathRAG-specific storage.
        
        Args:
            path: Path object to save
            embed: Whether to compute and store the path embedding
            
        Returns:
            Path to the saved file
        """
        # Create path directory if it doesn't exist
        os.makedirs(self.paths_dir, exist_ok=True)
        
        # Save path to JSON file
        filepath = os.path.join(self.paths_dir, f"{path.id}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(path.to_dict(), f, indent=2)
        
        # Add to index
        self.path_index.add_path(path)
        
        # Compute and store embedding if requested
        if embed:
            self.vector_store.get_path_embedding(path)
        
        logger.info(f"Saved path {path.id} to {filepath}")
        return filepath
    
    def load_path(self, path_id: str) -> Optional[RagPath]:
        """
        Load a path from storage.
        
        Args:
            path_id: ID of the path to load
            
        Returns:
            Path object or None if not found
        """
        # First check the index
        path = self.path_index.get_path(path_id)
        if path:
            return path
        
        # If not in index, try to load from file
        filepath = os.path.join(self.paths_dir, f"{path_id}.json")
        
        if not os.path.exists(filepath):
            logger.warning(f"Path {path_id} not found at {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                path_data = json.load(f)
            
            path = RagPath.from_dict(path_data)
            
            # Add to index
            self.path_index.add_path(path)
            
            logger.info(f"Loaded path {path_id} from {filepath}")
            return path
            
        except Exception as e:
            logger.error(f"Error loading path {path_id}: {e}")
            return None
    
    def delete_path(self, path_id: str) -> bool:
        """
        Delete a path from storage.
        
        Args:
            path_id: ID of the path to delete
            
        Returns:
            True if path was deleted, False if not found
        """
        # Remove from index
        if not self.path_index.remove_path(path_id):
            logger.warning(f"Path {path_id} not found in index")
        
        # Remove from vector store
        if path_id in self.vector_store.embeddings:
            del self.vector_store.embeddings[path_id]
        
        # Remove from file system
        filepath = os.path.join(self.paths_dir, f"{path_id}.json")
        
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Deleted path {path_id} from {filepath}")
            return True
        else:
            logger.warning(f"Path file {filepath} not found")
            return False
    
    def save_all_paths(self, paths: List[RagPath], embed: bool = True) -> None:
        """
        Save multiple paths to storage.
        
        Args:
            paths: List of Path objects to save
            embed: Whether to compute and store embeddings
        """
        for path in paths:
            self.save_path(path, embed=False)  # Don't embed individually
        
        # Batch embed if requested
        if embed:
            self.vector_store.embed_paths(paths)
        
        # Save the index and vector store
        self.save_index()
        self.save_vector_store()
    
    def load_all_paths(self) -> List[RagPath]:
        """
        Load all paths from storage.
        
        Returns:
            List of all Path objects
        """
        # Make sure the index is loaded
        self.load_index()
        
        return list(self.path_index.paths.values())
    
    def save_index(self) -> str:
        """
        Save the path index to disk.
        
        Returns:
            Path to the saved index file
        """
        os.makedirs(self.indexes_dir, exist_ok=True)
        filepath = os.path.join(self.indexes_dir, 'path_index.pkl')
        return self.path_index.save(filepath)
    
    def load_index(self) -> PathIndex:
        """
        Load the path index from disk.
        
        Returns:
            Loaded PathIndex object
        """
        filepath = os.path.join(self.indexes_dir, 'path_index.pkl')
        
        if os.path.exists(filepath):
            self.path_index.load(filepath)
            logger.info(f"Loaded path index from {filepath}")
        else:
            logger.warning(f"Path index file {filepath} not found")
        
        return self.path_index
    
    def save_vector_store(self) -> str:
        """
        Save the vector store to disk.
        
        Returns:
            Path to the saved vector store file
        """
        os.makedirs(self.embeddings_dir, exist_ok=True)
        filepath = os.path.join(self.embeddings_dir, 'path_embeddings.pkl')
        return self.vector_store.save(filepath)
    
    def load_vector_store(self) -> PathVectorStore:
        """
        Load the vector store from disk.
        
        Returns:
            Loaded PathVectorStore object
        """
        filepath = os.path.join(self.embeddings_dir, 'path_embeddings.pkl')
        
        if os.path.exists(filepath):
            self.vector_store.load(filepath)
            logger.info(f"Loaded vector store from {filepath}")
        else:
            logger.warning(f"Vector store file {filepath} not found")
        
        return self.vector_store
    
    def get_path_by_entity(self, entity_text: str, case_sensitive: bool = False) -> List[RagPath]:
        """
        Get paths containing a specific entity.
        
        Args:
            entity_text: Text of the entity to search for
            case_sensitive: Whether to perform case-sensitive matching
            
        Returns:
            List of Path objects
        """
        return self.path_index.get_paths_by_entity(entity_text, case_sensitive)
    
    def semantic_search(self, query: str, k: int = 10) -> List[Tuple[RagPath, float]]:
        """
        Search for paths semantically similar to a query.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (Path, similarity_score) tuples
        """
        # Get path IDs and scores from vector store
        path_ids_scores = self.vector_store.semantic_search(query, k)
        
        # Convert to Path objects
        results = []
        
        for path_id, score in path_ids_scores:
            path = self.path_index.get_path(path_id)
            if path:
                results.append((path, score))
        
        return results
    
    def advanced_search(self, 
                        entity_texts: Optional[List[str]] = None,
                        entity_types: Optional[List[str]] = None,
                        relationship_types: Optional[List[str]] = None,
                        document_ids: Optional[List[str]] = None,
                        min_length: Optional[int] = None,
                        max_length: Optional[int] = None,
                        min_score: Optional[float] = None,
                        query: Optional[str] = None,
                        k: int = 10) -> List[RagPath]:
        """
        Advanced search with multiple criteria.
        
        Args:
            entity_texts: List of entity texts to filter by
            entity_types: List of entity types to filter by
            relationship_types: List of relationship types to filter by
            document_ids: List of document IDs to filter by
            min_length: Minimum path length
            max_length: Maximum path length
            min_score: Minimum path score
            query: Optional semantic query text
            k: Maximum number of results to return
            
        Returns:
            List of Path objects matching all criteria
        """
        # Start with index-based filtering
        filtered_paths = self.path_index.advanced_search(
            entity_texts=entity_texts,
            entity_types=entity_types,
            relationship_types=relationship_types,
            document_ids=document_ids,
            min_length=min_length,
            max_length=max_length,
            min_score=min_score,
            k=k * 2  # Get more than needed for semantic reranking
        )
        
        # If no query, return filtered paths
        if not query:
            return filtered_paths[:k]
        
        # If query provided, rerank using semantic search
        if filtered_paths:
            # Get the query embedding
            query_embedding = self.vector_store._get_embedding(query)
            
            # Score each path
            scored_paths = []
            
            for path in filtered_paths:
                # Get or compute path embedding
                if path.id in self.vector_store.embeddings:
                    path_embedding = self.vector_store.embeddings[path.id].embedding
                else:
                    path_embedding = self.vector_store.get_path_embedding(path)
                
                # Calculate similarity
                similarity = np.dot(query_embedding, path_embedding)
                scored_paths.append((path, similarity))
            
            # Sort by similarity (descending)
            scored_paths.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k paths
            return [path for path, _ in scored_paths[:k]]
        
        # If no paths match filters, just do semantic search
        return [path for path, _ in self.semantic_search(query, k)]
    
    def check_document_universal_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Check if a document has universal metadata in the source_documents directory.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Universal metadata dictionary or None if not found
        """
        # Construct metadata file path
        metadata_path = FilePath(self.source_documents_dir) / f"{document_id}.json"
        
        if not metadata_path.exists():
            logger.warning(f"Universal metadata for document {document_id} not found at {metadata_path}")
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.info(f"Loaded universal metadata for document {document_id}")
            # Cast the result to the expected return type
            return cast(Dict[str, Any], metadata)
            
        except Exception as e:
            logger.error(f"Error loading universal metadata for document {document_id}: {e}")
            return None
    
    def get_paths_directory_size(self) -> int:
        """
        Get the size of the paths directory in bytes.
        
        Returns:
            Size in bytes
        """
        total_size = 0
        
        for dirpath, dirnames, filenames in os.walk(self.paths_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        return total_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the path storage.
        
        Returns:
            Dictionary of statistics
        """
        index_stats = self.path_index.get_stats()
        vector_store_stats = self.vector_store.get_stats()
        
        return {
            'total_paths': len(self.path_index.paths),
            'total_entities': index_stats['entity_count'],
            'total_relationship_types': index_stats['relationship_type_count'],
            'total_documents': index_stats['document_count'],
            'average_path_length': index_stats['average_path_length'],
            'embedding_dimension': vector_store_stats['dimension'],
            'embedding_model': vector_store_stats['model_name'],
            'storage_size_bytes': self.get_paths_directory_size()
        }
    
    def clear_all_data(self) -> None:
        """Clear all path data from storage."""
        # Clear in-memory data
        self.path_index.clear()
        self.vector_store.clear()
        
        # Clear files
        for directory in [self.paths_dir, self.indexes_dir, self.embeddings_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                os.makedirs(directory, exist_ok=True)
        
        logger.info("Cleared all path data from storage")
