"""
Path Vector Store for PathRAG

This module provides a vector store for path embeddings, enabling semantic search
over paths in the PathRAG implementation.
"""

import numpy as np
import os
import pickle
import time
import json
import logging
from typing import List, Dict, Any, Set, Tuple, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field

from .path_structures import Path, PathNode, PathEdge

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PathEmbedding:
    """
    Represents an embedding for a path with associated metadata.
    """
    path_id: str
    embedding: np.ndarray
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Ensure embedding is numpy array."""
        if not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding, dtype=np.float32)

class PathVectorStore:
    """
    Vector store for path embeddings, enabling semantic search over paths.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the path vector store with configuration.
        
        Args:
            config: Configuration dictionary with options for the vector store
        """
        self.config = config or {}
        
        # Embeddings storage
        self.embeddings: Dict[str, PathEmbedding] = {}
        
        # Storage directory
        self.storage_dir = self.config.get('storage_dir', None)
        
        # Embedding dimension
        self.dimension = self.config.get('dimension', 768)
        
        # Embedding model name/type
        self.model_name = self.config.get('model_name', 'default')
        
        # Whether to normalize embeddings
        self.normalize_embeddings = self.config.get('normalize_embeddings', True)
        
        # Batch size for processing
        self.batch_size = self.config.get('batch_size', 32)
        
        # Whether the embedding model is initialized
        self.model_initialized = False
        
        # The embedding model (set when initialize_model is called)
        self.model = None
        
        # Caching of similarity calculations for performance
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
    
    def initialize_model(self) -> None:
        """
        Initialize the embedding model.
        Supports different model backends based on configuration.
        """
        if self.model_initialized:
            return
        
        model_backend = self.config.get('model_backend', 'sentence_transformers')
        
        if model_backend == 'sentence_transformers':
            try:
                from sentence_transformers import SentenceTransformer
                
                model_name = self.config.get('model_name', 'all-MiniLM-L6-v2')
                self.model = SentenceTransformer(model_name)
                logger.info(f"Initialized SentenceTransformer model: {model_name}")
                
            except ImportError:
                logger.warning("SentenceTransformers not installed, using dummy embeddings")
                self.model = None
        
        elif model_backend == 'openai':
            try:
                import openai
                
                # Configure OpenAI client
                openai.api_key = self.config.get('openai_api_key', os.environ.get('OPENAI_API_KEY'))
                self.model = 'openai'  # Just a marker to indicate we're using OpenAI
                logger.info("Initialized OpenAI embedding model")
                
            except ImportError:
                logger.warning("OpenAI package not installed, using dummy embeddings")
                self.model = None
        
        elif model_backend == 'huggingface':
            try:
                from transformers import AutoModel, AutoTokenizer
                import torch
                
                model_name = self.config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                self.model = (model, tokenizer)
                logger.info(f"Initialized HuggingFace model: {model_name}")
                
            except ImportError:
                logger.warning("Transformers not installed, using dummy embeddings")
                self.model = None
        
        else:
            logger.warning(f"Unknown model backend: {model_backend}, using dummy embeddings")
            self.model = None
        
        self.model_initialized = True
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not self.model_initialized:
            self.initialize_model()
        
        if not text:
            # Return zero vector for empty text
            return np.zeros(self.dimension, dtype=np.float32)
        
        model_backend = self.config.get('model_backend', 'sentence_transformers')
        
        # Use appropriate backend for embedding
        if model_backend == 'sentence_transformers' and self.model:
            # Use SentenceTransformer
            embedding = self.model.encode(text, convert_to_numpy=True)
            
        elif model_backend == 'openai' and self.model:
            # Use OpenAI API
            import openai
            response = openai.Embedding.create(
                input=text,
                model=self.config.get('openai_model', 'text-embedding-ada-002')
            )
            embedding = np.array(response['data'][0]['embedding'], dtype=np.float32)
            
        elif model_backend == 'huggingface' and self.model:
            # Use HuggingFace model
            import torch
            model, tokenizer = self.model
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Use mean pooling of last hidden state
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            masked_embeddings = embeddings * mask
            summed = torch.sum(masked_embeddings, 1)
            summed_mask = torch.sum(mask, 1)
            mean_pooled = summed / summed_mask
            embedding = mean_pooled.cpu().numpy()[0]
            
        else:
            # Fallback to random embedding if no model available
            logger.warning("No embedding model available, using random embedding")
            embedding = np.random.randn(self.dimension).astype(np.float32)
        
        # Normalize if configured
        if self.normalize_embeddings:
            embedding = self._normalize_embedding(embedding)
        
        return embedding
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding vector to unit length.
        
        Args:
            embedding: Embedding vector
            
        Returns:
            Normalized embedding vector
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def get_path_embedding(self, path: Path) -> np.ndarray:
        """
        Get embedding for a path.
        
        Args:
            path: Path object to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Check if embedding already exists
        if path.id in self.embeddings:
            return self.embeddings[path.id].embedding
        
        # Extract text representation of the path
        path_text = self._path_to_text(path)
        
        # Get embedding
        embedding = self._get_embedding(path_text)
        
        # Store embedding
        self.add_embedding(path.id, embedding)
        
        return embedding
    
    def _path_to_text(self, path: Path) -> str:
        """
        Convert a path to a text representation for embedding.
        
        Args:
            path: Path object to convert
            
        Returns:
            Text representation of the path
        """
        # Join node texts with relationship types
        texts = []
        
        for i, node in enumerate(path.nodes):
            texts.append(node.text)
            
            # Add relationship if not the last node
            if i < len(path.edges):
                edge = path.edges[i]
                texts.append(f"({edge.type})")
        
        return " ".join(texts)
    
    def add_embedding(self, path_id: str, embedding: np.ndarray) -> None:
        """
        Add an embedding for a path.
        
        Args:
            path_id: ID of the path
            embedding: Embedding vector
        """
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Normalize if configured
        if self.normalize_embeddings:
            embedding = self._normalize_embedding(embedding)
        
        self.embeddings[path_id] = PathEmbedding(path_id=path_id, embedding=embedding)
        
        # Clear cache entries involving this path
        self.similarity_cache = {
            k: v for k, v in self.similarity_cache.items() 
            if k[0] != path_id and k[1] != path_id
        }
    
    def embed_paths(self, paths: List[Path]) -> Dict[str, np.ndarray]:
        """
        Embed multiple paths.
        
        Args:
            paths: List of Path objects to embed
            
        Returns:
            Dictionary mapping path IDs to embeddings
        """
        result = {}
        
        # Process in batches
        for i in range(0, len(paths), self.batch_size):
            batch = paths[i:i+self.batch_size]
            
            # Get text representations
            texts = [self._path_to_text(path) for path in batch]
            
            # Get embeddings
            model_backend = self.config.get('model_backend', 'sentence_transformers')
            
            if model_backend == 'sentence_transformers' and self.model:
                # Batch encode with SentenceTransformer
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                
                # Store embeddings
                for j, path in enumerate(batch):
                    embedding = embeddings[j]
                    if self.normalize_embeddings:
                        embedding = self._normalize_embedding(embedding)
                    self.add_embedding(path.id, embedding)
                    result[path.id] = embedding
            
            else:
                # Fallback to individual encoding
                for path in batch:
                    embedding = self.get_path_embedding(path)
                    result[path.id] = embedding
        
        return result
    
    def similarity(self, path_id1: str, path_id2: str) -> float:
        """
        Calculate similarity between two paths by their embeddings.
        
        Args:
            path_id1: ID of first path
            path_id2: ID of second path
            
        Returns:
            Cosine similarity score
        """
        # Check cache
        cache_key = (path_id1, path_id2)
        reverse_cache_key = (path_id2, path_id1)
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        if reverse_cache_key in self.similarity_cache:
            return self.similarity_cache[reverse_cache_key]
        
        # Make sure both embeddings exist
        if path_id1 not in self.embeddings or path_id2 not in self.embeddings:
            return 0.0
        
        # Get embeddings
        emb1 = self.embeddings[path_id1].embedding
        emb2 = self.embeddings[path_id2].embedding
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2)
        
        # Store in cache
        self.similarity_cache[cache_key] = similarity
        
        return similarity
    
    def semantic_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for paths semantically similar to a query.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (path_id, similarity_score) tuples
        """
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Calculate similarities
        similarities = []
        
        for path_id, path_embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, path_embedding.embedding)
            similarities.append((path_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return similarities[:k]
    
    def similar_paths(self, path_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Find paths similar to a given path.
        
        Args:
            path_id: ID of the reference path
            k: Number of results to return
            
        Returns:
            List of (path_id, similarity_score) tuples
        """
        if path_id not in self.embeddings:
            return []
        
        reference_embedding = self.embeddings[path_id].embedding
        
        # Calculate similarities
        similarities = []
        
        for other_id, other_embedding in self.embeddings.items():
            if other_id == path_id:
                continue
            
            similarity = np.dot(reference_embedding, other_embedding.embedding)
            similarities.append((other_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return similarities[:k]
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the vector store to a file.
        
        Args:
            filepath: Path to save to (if None, uses storage_dir/path_embeddings.pkl)
            
        Returns:
            Path to the saved file
        """
        if filepath is None:
            if self.storage_dir is None:
                raise ValueError("No storage directory specified")
            
            # Ensure directory exists
            os.makedirs(self.storage_dir, exist_ok=True)
            filepath = os.path.join(self.storage_dir, "path_embeddings.pkl")
        
        # Prepare serializable data
        serialized_data = {
            'embeddings': {
                path_id: {
                    'path_id': emb.path_id,
                    'embedding': emb.embedding.tolist(),
                    'timestamp': emb.timestamp
                }
                for path_id, emb in self.embeddings.items()
            },
            'config': self.config,
            'dimension': self.dimension,
            'model_name': self.model_name,
            'normalize_embeddings': self.normalize_embeddings
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(serialized_data, f)
        
        logger.info(f"Saved path embeddings to {filepath}")
        return filepath
    
    def load(self, filepath: str) -> 'PathVectorStore':
        """
        Load the vector store from a file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            Loaded PathVectorStore
        """
        with open(filepath, 'rb') as f:
            serialized_data = pickle.load(f)
        
        # Load config
        self.config = serialized_data.get('config', {})
        self.dimension = serialized_data.get('dimension', self.dimension)
        self.model_name = serialized_data.get('model_name', self.model_name)
        self.normalize_embeddings = serialized_data.get('normalize_embeddings', self.normalize_embeddings)
        
        # Load embeddings
        self.embeddings = {}
        
        for path_id, emb_data in serialized_data['embeddings'].items():
            embedding = np.array(emb_data['embedding'], dtype=np.float32)
            self.embeddings[path_id] = PathEmbedding(
                path_id=emb_data['path_id'],
                embedding=embedding,
                timestamp=emb_data.get('timestamp', time.time())
            )
        
        logger.info(f"Loaded {len(self.embeddings)} path embeddings from {filepath}")
        return self
    
    def clear(self) -> None:
        """Clear all embeddings from the store."""
        self.embeddings.clear()
        self.similarity_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'embedding_count': len(self.embeddings),
            'dimension': self.dimension,
            'model_name': self.model_name,
            'normalize_embeddings': self.normalize_embeddings,
            'cache_size': len(self.similarity_cache)
        }
