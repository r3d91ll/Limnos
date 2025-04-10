"""
Performance Metrics Collection Utilities

This module provides utilities for collecting and analyzing performance metrics
for both PathRAG and GraphRAG frameworks.
"""

import time
import json
import os
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic, Collection, cast
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
import tracemalloc
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class TimingMetric:
    """Timing metric for a specific operation."""
    operation: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    
    def complete(self) -> float:
        """
        Complete the timing metric and calculate duration.
        
        Returns:
            Duration in milliseconds
        """
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        return self.duration_ms

@dataclass
class MemoryMetric:
    """Memory usage metric for a specific operation."""
    operation: str
    start_memory_bytes: int
    peak_memory_bytes: int = 0
    end_memory_bytes: int = 0
    difference_bytes: int = 0
    
    def complete(self, peak_memory: int) -> int:
        """
        Complete the memory metric and calculate difference.
        
        Args:
            peak_memory: Peak memory usage in bytes
            
        Returns:
            Memory difference in bytes
        """
        self.end_memory_bytes = psutil.Process().memory_info().rss
        self.peak_memory_bytes = peak_memory
        self.difference_bytes = self.end_memory_bytes - self.start_memory_bytes
        return self.difference_bytes

@dataclass
class StorageMetric:
    """Storage size metric for a specific component."""
    component: str
    path: str
    size_bytes: int = 0
    
    def measure(self) -> int:
        """
        Measure the storage size of the component.
        
        Returns:
            Size in bytes
        """
        if not os.path.exists(self.path):
            self.size_bytes = 0
            return 0
            
        if os.path.isdir(self.path):
            total_size = 0
            for dirpath, _, filenames in os.walk(self.path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
            self.size_bytes = total_size
        else:
            self.size_bytes = os.path.getsize(self.path)
            
        return self.size_bytes

@dataclass
class MetricsCollection:
    """Collection of metrics for a framework test run."""
    framework: str
    test_name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    timing_metrics: Dict[str, TimingMetric] = field(default_factory=dict)
    memory_metrics: Dict[str, MemoryMetric] = field(default_factory=dict)
    storage_metrics: Dict[str, StorageMetric] = field(default_factory=dict)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self) -> None:
        """Complete the metrics collection."""
        self.end_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics collection to dictionary."""
        result = {
            'framework': self.framework,
            'test_name': self.test_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'timing_metrics': {k: asdict(v) for k, v in self.timing_metrics.items()},
            'memory_metrics': {k: asdict(v) for k, v in self.memory_metrics.items()},
            'storage_metrics': {k: asdict(v) for k, v in self.storage_metrics.items()},
            'additional_metrics': self.additional_metrics
        }
        return result
    
    def save(self, filepath: str) -> None:
        """
        Save metrics to a JSON file.
        
        Args:
            filepath: Path to save the metrics
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'MetricsCollection':
        """
        Load metrics from a JSON file.
        
        Args:
            filepath: Path to load the metrics from
            
        Returns:
            Loaded metrics collection
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        collection = cls(
            framework=data['framework'],
            test_name=data['test_name'],
            start_time=datetime.fromisoformat(data['start_time'])
        )
        
        if data['end_time']:
            collection.end_time = datetime.fromisoformat(data['end_time'])
        
        # Load timing metrics
        for k, v in data['timing_metrics'].items():
            timing_metric = TimingMetric(
                operation=v['operation'],
                start_time=v['start_time'],
                end_time=v['end_time'],
                duration_ms=v['duration_ms']
            )
            collection.timing_metrics[k] = timing_metric
        
        # Load memory metrics
        for k, v in data['memory_metrics'].items():
            memory_metric = MemoryMetric(
                operation=v['operation'],
                start_memory_bytes=v['start_memory_bytes'],
                peak_memory_bytes=v['peak_memory_bytes'],
                end_memory_bytes=v['end_memory_bytes'],
                difference_bytes=v['difference_bytes']
            )
            collection.memory_metrics[k] = memory_metric
        
        # Load storage metrics
        for k, v in data['storage_metrics'].items():
            storage_metric = StorageMetric(
                component=v['component'],
                path=v['path'],
                size_bytes=v['size_bytes']
            )
            collection.storage_metrics[k] = storage_metric
        
        # Load additional metrics
        collection.additional_metrics = data['additional_metrics']
        
        return collection


class PerformanceTracker:
    """Utility for tracking performance metrics during testing."""
    
    def __init__(self, framework: str, test_name: str):
        """
        Initialize the performance tracker.
        
        Args:
            framework: Name of the framework being tested
            test_name: Name of the test being run
        """
        self.framework = framework
        self.test_name = test_name
        self.metrics = MetricsCollection(framework=framework, test_name=test_name)
        self.active_timing: Dict[str, TimingMetric] = {}
        self.active_memory: Dict[str, MemoryMetric] = {}
        self.trace_memory = False
        
    def start_timing(self, operation: str) -> TimingMetric:
        """
        Start timing an operation.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Timing metric object
        """
        metric = TimingMetric(operation=operation, start_time=time.time())
        self.active_timing[operation] = metric
        return metric
    
    def end_timing(self, operation: str) -> float:
        """
        End timing an operation and record the metric.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Duration in milliseconds
        """
        if operation not in self.active_timing:
            logger.warning(f"No active timing for operation: {operation}")
            return 0.0
        
        metric = self.active_timing.pop(operation)
        duration = metric.complete()
        self.metrics.timing_metrics[operation] = metric
        return float(duration)
    
    def start_memory_tracking(self, operation: str) -> MemoryMetric:
        """
        Start tracking memory usage for an operation.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Memory metric object
        """
        if not self.trace_memory:
            tracemalloc.start()
            self.trace_memory = True
        
        metric = MemoryMetric(
            operation=operation,
            start_memory_bytes=psutil.Process().memory_info().rss
        )
        self.active_memory[operation] = metric
        return metric
    
    def end_memory_tracking(self, operation: str) -> int:
        """
        End tracking memory usage for an operation and record the metric.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Memory difference in bytes
        """
        if operation not in self.active_memory:
            logger.warning(f"No active memory tracking for operation: {operation}")
            return 0
        
        peak_memory = 0
        if self.trace_memory:
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.trace_memory = False
        
        metric = self.active_memory.pop(operation)
        difference = metric.complete(peak_memory)
        self.metrics.memory_metrics[operation] = metric
        return int(difference)
    
    def measure_storage(self, component: str, path: str) -> int:
        """
        Measure storage size for a component.
        
        Args:
            component: Name of the component
            path: Path to measure
            
        Returns:
            Size in bytes
        """
        metric = StorageMetric(component=component, path=path)
        size = metric.measure()
        self.metrics.storage_metrics[component] = metric
        return size
    
    def add_metric(self, name: str, value: Any) -> None:
        """
        Add an additional metric.
        
        Args:
            name: Name of the metric
            value: Value of the metric
        """
        self.metrics.additional_metrics[name] = value
    
    def complete(self) -> MetricsCollection:
        """
        Complete all metrics collection.
        
        Returns:
            Completed metrics collection
        """
        # End any active timing operations
        for operation in list(self.active_timing.keys()):
            self.end_timing(operation)
        
        # End any active memory tracking
        for operation in list(self.active_memory.keys()):
            self.end_memory_tracking(operation)
        
        # Complete the metrics collection
        self.metrics.complete()
        return self.metrics
    
    def save_metrics(self, filepath: str) -> None:
        """
        Save metrics to a file.
        
        Args:
            filepath: Path to save the metrics
        """
        self.complete()
        self.metrics.save(filepath)
    
    @contextmanager
    def time_operation(self, operation: str):
        """
        Context manager for timing an operation.
        
        Args:
            operation: Name of the operation
        """
        self.start_timing(operation)
        try:
            yield
        finally:
            self.end_timing(operation)
    
    @contextmanager
    def track_memory(self, operation: str):
        """
        Context manager for tracking memory usage.
        
        Args:
            operation: Name of the operation
        """
        self.start_memory_tracking(operation)
        try:
            yield
        finally:
            self.end_memory_tracking(operation)


# Comparison utilities
def compare_frameworks(pathrag_metrics: MetricsCollection, graphrag_metrics: MetricsCollection) -> Dict[str, Any]:
    """
    Compare metrics between PathRAG and GraphRAG frameworks.
    
    Args:
        pathrag_metrics: Metrics for PathRAG
        graphrag_metrics: Metrics for GraphRAG
        
    Returns:
        Dictionary of comparison results
    """
    comparison = {
        'test_name': f"Comparison: {pathrag_metrics.test_name} vs {graphrag_metrics.test_name}",
        'timestamp': datetime.now().isoformat(),
        'timing_comparison': {},
        'memory_comparison': {},
        'storage_comparison': {},
        'additional_comparison': {}
    }
    
    # Compare timing metrics
    all_timing_operations = set(pathrag_metrics.timing_metrics.keys()) | set(graphrag_metrics.timing_metrics.keys())
    for operation in all_timing_operations:
        pathrag_duration = pathrag_metrics.timing_metrics.get(operation, TimingMetric(operation=operation, start_time=0)).duration_ms
        graphrag_duration = graphrag_metrics.timing_metrics.get(operation, TimingMetric(operation=operation, start_time=0)).duration_ms
        
        if 'timing_comparison' not in comparison:
            comparison['timing_comparison'] = {}
        # Cast to avoid Collection[str] type issues
        timing_dict = cast(Dict[str, Dict[str, Any]], comparison['timing_comparison'])
        timing_dict[operation] = {
            'pathrag_ms': pathrag_duration,
            'graphrag_ms': graphrag_duration,
            'difference_ms': pathrag_duration - graphrag_duration,
            'ratio': pathrag_duration / graphrag_duration if graphrag_duration > 0 else float('inf')
        }
    
    # Compare memory metrics
    all_memory_operations = set(pathrag_metrics.memory_metrics.keys()) | set(graphrag_metrics.memory_metrics.keys())
    for operation in all_memory_operations:
        pathrag_memory = pathrag_metrics.memory_metrics.get(operation, MemoryMetric(operation=operation, start_memory_bytes=0)).difference_bytes
        graphrag_memory = graphrag_metrics.memory_metrics.get(operation, MemoryMetric(operation=operation, start_memory_bytes=0)).difference_bytes
        
        if 'memory_comparison' not in comparison:
            comparison['memory_comparison'] = {}
        # Cast to avoid Collection[str] type issues
        memory_dict = cast(Dict[str, Dict[str, Any]], comparison['memory_comparison'])
        memory_dict[operation] = {
            'pathrag_bytes': pathrag_memory,
            'graphrag_bytes': graphrag_memory,
            'difference_bytes': pathrag_memory - graphrag_memory,
            'ratio': pathrag_memory / graphrag_memory if graphrag_memory > 0 else float('inf')
        }
    
    # Compare storage metrics
    all_components = set(pathrag_metrics.storage_metrics.keys()) | set(graphrag_metrics.storage_metrics.keys())
    for component in all_components:
        pathrag_size = pathrag_metrics.storage_metrics.get(component, StorageMetric(component=component, path="")).size_bytes
        graphrag_size = graphrag_metrics.storage_metrics.get(component, StorageMetric(component=component, path="")).size_bytes
        
        if 'storage_comparison' not in comparison:
            comparison['storage_comparison'] = {}
        # Cast to avoid Collection[str] type issues
        storage_dict = cast(Dict[str, Dict[str, Any]], comparison['storage_comparison'])
        storage_dict[component] = {
            'pathrag_bytes': pathrag_size,
            'graphrag_bytes': graphrag_size,
            'difference_bytes': pathrag_size - graphrag_size,
            'ratio': pathrag_size / graphrag_size if graphrag_size > 0 else float('inf')
        }
    
    # Compare additional metrics where possible
    all_additional = set(pathrag_metrics.additional_metrics.keys()) | set(graphrag_metrics.additional_metrics.keys())
    for metric in all_additional:
        pathrag_value = pathrag_metrics.additional_metrics.get(metric)
        graphrag_value = graphrag_metrics.additional_metrics.get(metric)
        
        if 'additional_comparison' not in comparison:
            comparison['additional_comparison'] = {}
            
        if pathrag_value is not None and graphrag_value is not None and isinstance(pathrag_value, (int, float)) and isinstance(graphrag_value, (int, float)):
            # Cast to avoid Collection[str] type issues
            additional_dict = cast(Dict[str, Dict[str, Any]], comparison['additional_comparison'])
            additional_dict[metric] = {
                'pathrag': pathrag_value,
                'graphrag': graphrag_value,
                'difference': pathrag_value - graphrag_value,
                'ratio': pathrag_value / graphrag_value if graphrag_value != 0 else float('inf')
            }
        else:
            if 'additional_comparison' not in comparison:
                comparison['additional_comparison'] = {}
            # Cast to avoid Collection[str] type issues
            additional_dict = cast(Dict[str, Dict[str, Any]], comparison['additional_comparison'])
            additional_dict[metric] = {
                'pathrag': pathrag_value,
                'graphrag': graphrag_value,
                'comparison': 'Non-numeric values, cannot calculate difference/ratio'
            }
    
    return comparison

def save_comparison(comparison: Dict[str, Any], filepath: str) -> None:
    """
    Save comparison results to a file.
    
    Args:
        comparison: Comparison results
        filepath: Path to save the results
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
