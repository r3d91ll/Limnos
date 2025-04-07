"""
Batch Operations

Provides efficient batch processing capabilities for graph storage operations.
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import networkx as nx
import redis
from redis.client import Pipeline

logger = logging.getLogger(__name__)

class BatchOperations:
    """
    Enables efficient batch operations for storing and retrieving multiple graphs.
    
    This class optimizes Redis operations by batching multiple requests together,
    reducing network overhead and improving performance.
    """
    
    def __init__(self, 
                 redis_graph_manager,
                 batch_size: int = 100,
                 auto_execute: bool = True):
        """
        Initialize batch operations handler.
        
        Args:
            redis_graph_manager: RedisGraphManager instance
            batch_size: Maximum number of operations per batch
            auto_execute: Whether to automatically execute when batch_size is reached
        """
        self.redis_graph_manager = redis_graph_manager
        self.batch_size = batch_size
        self.auto_execute = auto_execute
        
        # Current batch of operations
        self.current_batch: List[Dict[str, Any]] = []
        
        # Lock for thread safety
        self.batch_lock = threading.RLock()
        
        # Stats
        self.stats = {
            'batches_executed': 0,
            'operations_processed': 0,
            'last_execution_time': 0.0
        }
        
        logger.info(f"Initialized BatchOperations with batch_size={batch_size}, auto_execute={auto_execute}")
    
    def store_graph(self, graph_id: str, graph: nx.Graph, metadata: Dict[str, Any] = None) -> None:
        """
        Add a graph storage operation to the batch.
        
        Args:
            graph_id: Unique identifier for the graph
            graph: NetworkX graph object
            metadata: Optional metadata dict to store with the graph
        """
        with self.batch_lock:
            operation = {
                'type': 'store',
                'graph_id': graph_id,
                'graph': graph,
                'metadata': metadata
            }
            self.current_batch.append(operation)
            
            if self.auto_execute and len(self.current_batch) >= self.batch_size:
                self.execute_batch()
    
    def delete_graph(self, graph_id: str) -> None:
        """
        Add a graph deletion operation to the batch.
        
        Args:
            graph_id: Unique identifier for the graph to delete
        """
        with self.batch_lock:
            operation = {
                'type': 'delete',
                'graph_id': graph_id
            }
            self.current_batch.append(operation)
            
            if self.auto_execute and len(self.current_batch) >= self.batch_size:
                self.execute_batch()
    
    def batch_size_reached(self) -> bool:
        """Check if current batch has reached maximum size."""
        with self.batch_lock:
            return len(self.current_batch) >= self.batch_size
    
    def current_batch_size(self) -> int:
        """Get the current batch size."""
        with self.batch_lock:
            return len(self.current_batch)
    
    def execute_batch(self) -> Dict[str, Any]:
        """
        Execute the current batch of operations.
        
        Returns:
            Dict[str, Any]: Batch execution results
        """
        with self.batch_lock:
            if not self.current_batch:
                logger.debug("No operations in current batch to execute")
                return {'success': True, 'operations_processed': 0, 'errors': []}
            
            start_time = time.time()
            operations_count = len(self.current_batch)
            
            # Process batch operations
            results = self._process_batch()
            
            # Clear the current batch
            self.current_batch = []
            
            # Update stats
            execution_time = time.time() - start_time
            self.stats['batches_executed'] += 1
            self.stats['operations_processed'] += operations_count
            self.stats['last_execution_time'] = execution_time
            
            logger.info(f"Executed batch with {operations_count} operations in {execution_time:.4f}s")
            
            return results
    
    def _process_batch(self) -> Dict[str, Any]:
        """
        Process all operations in the current batch.
        
        Returns:
            Dict[str, Any]: Execution results
        """
        # Get serializer for graph serialization
        from .graph_serializer import GraphSerializer
        serializer = GraphSerializer()
        
        # Get Redis client and prepare pipeline
        redis_client = self.redis_graph_manager.redis_client
        pipeline = redis_client.pipeline(transaction=False)
        
        # Track which operations are being executed
        store_operations = []
        delete_operations = []
        errors = []
        
        # Prepare all operations
        try:
            for operation in self.current_batch:
                op_type = operation['type']
                graph_id = operation['graph_id']
                
                if op_type == 'store':
                    # Serialize graph
                    graph = operation['graph']
                    metadata = operation.get('metadata')
                    
                    try:
                        # Serialize graph
                        graph_data = serializer.serialize(graph)
                        
                        # Add to pipeline
                        graph_key = self.redis_graph_manager.graph_key(graph_id)
                        pipeline.set(graph_key, graph_data)
                        pipeline.expire(graph_key, self.redis_graph_manager.ttl)
                        
                        # Add metadata if provided
                        if metadata:
                            metadata_key = self.redis_graph_manager.metadata_key(graph_id)
                            pipeline.hset(metadata_key, mapping=metadata)
                            pipeline.expire(metadata_key, self.redis_graph_manager.ttl)
                        
                        store_operations.append(graph_id)
                    except Exception as e:
                        errors.append({
                            'operation': 'store',
                            'graph_id': graph_id,
                            'error': str(e)
                        })
                        logger.error(f"Error preparing store operation for graph '{graph_id}': {e}")
                
                elif op_type == 'delete':
                    try:
                        # Add deletion to pipeline
                        graph_key = self.redis_graph_manager.graph_key(graph_id)
                        metadata_key = self.redis_graph_manager.metadata_key(graph_id)
                        
                        pipeline.delete(graph_key)
                        pipeline.delete(metadata_key)
                        
                        delete_operations.append(graph_id)
                    except Exception as e:
                        errors.append({
                            'operation': 'delete',
                            'graph_id': graph_id,
                            'error': str(e)
                        })
                        logger.error(f"Error preparing delete operation for graph '{graph_id}': {e}")
            
            # Execute pipeline if there are operations
            if store_operations or delete_operations:
                pipeline_results = pipeline.execute()
                logger.debug(f"Pipeline execution complete: {len(pipeline_results)} Redis commands executed")
        
        except Exception as e:
            logger.error(f"Error executing batch operations: {e}")
            errors.append({
                'operation': 'batch',
                'error': str(e)
            })
        
        # Prepare results
        return {
            'success': len(errors) == 0,
            'operations_processed': len(store_operations) + len(delete_operations),
            'stored': store_operations,
            'deleted': delete_operations,
            'errors': errors
        }
    
    def flush(self) -> Dict[str, Any]:
        """
        Flush any pending operations by executing the current batch.
        
        Returns:
            Dict[str, Any]: Batch execution results
        """
        return self.execute_batch()
    
    def store_graphs_bulk(self, 
                       graphs: Dict[str, nx.Graph], 
                       metadata: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store multiple graphs in a single batch operation.
        
        Args:
            graphs: Dictionary of graph_id -> graph_object
            metadata: Optional dictionary of graph_id -> metadata
            
        Returns:
            Dict[str, Any]: Batch execution results
        """
        with self.batch_lock:
            # Clear current batch to start fresh
            self.current_batch = []
            
            # Add all graphs to batch
            for graph_id, graph in graphs.items():
                graph_metadata = None
                if metadata and graph_id in metadata:
                    graph_metadata = metadata[graph_id]
                
                self.current_batch.append({
                    'type': 'store',
                    'graph_id': graph_id,
                    'graph': graph,
                    'metadata': graph_metadata
                })
            
            # Execute batch
            return self.execute_batch()
    
    def delete_graphs_bulk(self, graph_ids: List[str]) -> Dict[str, Any]:
        """
        Delete multiple graphs in a single batch operation.
        
        Args:
            graph_ids: List of graph IDs to delete
            
        Returns:
            Dict[str, Any]: Batch execution results
        """
        with self.batch_lock:
            # Clear current batch to start fresh
            self.current_batch = []
            
            # Add all deletions to batch
            for graph_id in graph_ids:
                self.current_batch.append({
                    'type': 'delete',
                    'graph_id': graph_id
                })
            
            # Execute batch
            return self.execute_batch()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get batch operation statistics.
        
        Returns:
            Dict[str, Any]: Statistics about batch operations
        """
        with self.batch_lock:
            stats = dict(self.stats)
            stats['current_batch_size'] = len(self.current_batch)
            return stats
