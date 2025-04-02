"""
Dataset Loading Utilities

This module provides utilities for loading and processing datasets used in the experiments.
"""

import json
import os
from typing import Dict, List, Any, Iterator, Optional


class DatasetLoader:
    """Class for loading and processing experimental datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset loader.
        
        Args:
            config: Configuration dictionary with dataset paths
        """
        self.config = config
        self.dataset_paths = config.get("dataset_paths", {})
        self.base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "datasets"
        )
    
    def get_dataset_path(self, dataset_name: str) -> str:
        """
        Get the full path to a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Full path to the dataset file
        """
        if dataset_name in self.dataset_paths:
            path = self.dataset_paths[dataset_name]
            # If path is relative, make it absolute
            if not os.path.isabs(path):
                path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    path
                )
            return path
        else:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
    
    def load_kilt_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Load a KILT format dataset.
        
        Args:
            dataset_name: Name of the KILT dataset (e.g., 'kilt_nq', 'kilt_hotpotqa')
            
        Returns:
            List of examples from the dataset
        """
        path = self.get_dataset_path(dataset_name)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    example = json.loads(line)
                    examples.append(example)
        
        return examples
    
    def kilt_dataset_iter(self, dataset_name: str) -> Iterator[Dict[str, Any]]:
        """
        Iterate through a KILT format dataset without loading it all into memory.
        
        Args:
            dataset_name: Name of the KILT dataset
            
        Yields:
            Each example from the dataset
        """
        path = self.get_dataset_path(dataset_name)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    example = json.loads(line)
                    yield example
    
    def extract_query_answer_pairs(self, dataset_name: str) -> List[Dict[str, str]]:
        """
        Extract query-answer pairs from a KILT dataset.
        
        Args:
            dataset_name: Name of the KILT dataset
            
        Returns:
            List of dictionaries with 'query' and 'answer' keys
        """
        examples = self.load_kilt_dataset(dataset_name)
        pairs = []
        
        for example in examples:
            query = example.get('input', '')
            
            # Extract answer text from KILT format
            answers = []
            if 'output' in example and isinstance(example['output'], list):
                for output in example['output']:
                    if 'answer' in output:
                        answers.append(output['answer'])
            
            # Join multiple answers if present
            answer = ' '.join(answers) if answers else ''
            
            pairs.append({
                'query': query,
                'answer': answer,
                'id': example.get('id', ''),
                'provenance': example.get('output', [{}])[0].get('provenance', [])
                if example.get('output') else []
            })
        
        return pairs
