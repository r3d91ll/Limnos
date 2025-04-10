#!/usr/bin/env python3
"""
Framework Testing Runner

This script provides a command-line interface for running tests on both
PathRAG and GraphRAG frameworks and comparing the results.
"""

# mypy: disable-error-code="import-untyped"
# mypy: disable-error-code="no-untyped-def"
import os
import sys
import argparse
import logging
import json
from pathlib import Path
import importlib
from typing import Dict, List, Any, Optional, Tuple

# Ensure the parent directory is in path so we can import module files
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from our framework testing modules
from experiments.framework_testing.common.metrics import (
    MetricsCollection, 
    compare_frameworks, 
    save_comparison
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_results_dir() -> Tuple[Path, Path, Path]:
    """
    Setup directories for test results.
    
    Returns:
        Tuple of (results_dir, pathrag_results_dir, graphrag_results_dir)
    """
    results_dir = Path(__file__).parent / "results"
    pathrag_results_dir = results_dir / "pathrag"
    graphrag_results_dir = results_dir / "graphrag"
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(pathrag_results_dir, exist_ok=True)
    os.makedirs(graphrag_results_dir, exist_ok=True)
    
    return results_dir, pathrag_results_dir, graphrag_results_dir

def run_embedding_tests(framework: str) -> Dict[str, Any]:
    """
    Run embedding tests for the specified framework.
    
    Args:
        framework: Either 'pathrag' or 'graphrag'
        
    Returns:
        Dictionary of test results
    """
    logger.info(f"Running embedding tests for {framework}")
    
    try:
        # Use relative imports instead of file loading to avoid module conflicts
        if framework.lower() == 'pathrag':
            # Import using package structure with relative imports
            from .pathrag.test_embedding import main as run_tests
        elif framework.lower() == 'graphrag':
            # Import using package structure with relative imports
            from .graphrag.test_embedding import main as run_tests
        else:
            raise ValueError(f"Unknown framework: {framework}")
        
        # Get results and ensure it's a dictionary
        results = run_tests()
        if not isinstance(results, dict):
            results = {"error": "Test did not return a dictionary"}
        
        # The function is annotated to return Dict[str, Any], and we've verified
        # results is a dict, so this satisfies the type checker
        return results
    
    except Exception as e:
        logger.error(f"Error running embedding tests for {framework}: {e}")
        return {"error": str(e)}

def run_path_construction_tests(framework: str) -> Dict[str, Any]:
    """
    Run path construction tests for PathRAG or graph construction tests for GraphRAG.
    
    Args:
        framework: Either 'pathrag' or 'graphrag'
        
    Returns:
        Dictionary of test results
    """
    logger.info(f"Running {'path' if framework == 'pathrag' else 'graph'} construction tests for {framework}")
    
    # This would import the appropriate module once implemented
    logger.info(f"Path/graph construction tests not yet implemented for {framework}")
    return {"status": "not_implemented"}

def run_query_tests(framework: str) -> Dict[str, Any]:
    """
    Run query processing tests for the specified framework.
    
    Args:
        framework: Either 'pathrag' or 'graphrag'
        
    Returns:
        Dictionary of test results
    """
    logger.info(f"Running query tests for {framework}")
    
    # This would import the appropriate module once implemented
    logger.info(f"Query tests not yet implemented for {framework}")
    return {"status": "not_implemented"}

def compare_test_results(test_type: str) -> Dict[str, Any]:
    """
    Compare test results between PathRAG and GraphRAG.
    
    Args:
        test_type: Type of test to compare (e.g., 'embedding')
        
    Returns:
        Dictionary of comparison results
    """
    logger.info(f"Comparing {test_type} test results")
    
    results_dir, pathrag_dir, graphrag_dir = setup_results_dir()
    
    try:
        # Load PathRAG metrics
        pathrag_files = list(pathrag_dir.glob(f"{test_type}*.json"))
        if not pathrag_files:
            raise FileNotFoundError(f"No {test_type} test results found for PathRAG")
        
        # Load GraphRAG metrics
        graphrag_files = list(graphrag_dir.glob(f"{test_type}*.json"))
        if not graphrag_files:
            raise FileNotFoundError(f"No {test_type} test results found for GraphRAG")
        
        # Compare each pair of matching files
        comparisons = {}
        
        for pathrag_file in pathrag_files:
            # Find matching GraphRAG file
            graphrag_file = next(
                (f for f in graphrag_files if f.name == pathrag_file.name), 
                None
            )
            
            if graphrag_file:
                # Load metrics
                pathrag_metrics = MetricsCollection.load(str(pathrag_file))
                graphrag_metrics = MetricsCollection.load(str(graphrag_file))
                
                # Compare
                comparison = compare_frameworks(pathrag_metrics, graphrag_metrics)
                
                # Save comparison
                comparison_path = results_dir / f"comparison_{pathrag_file.stem}.json"
                save_comparison(comparison, str(comparison_path))
                
                comparisons[pathrag_file.stem] = comparison
        
        return comparisons
    
    except Exception as e:
        logger.error(f"Error comparing {test_type} test results: {e}")
        return {"error": str(e)}

def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Framework Testing Runner")
    
    parser.add_argument(
        "--framework", 
        choices=["pathrag", "graphrag", "both"], 
        default="both",
        help="Framework to test (default: both)"
    )
    
    parser.add_argument(
        "--test", 
        choices=["embedding", "path", "query", "all"], 
        default="all",
        help="Test to run (default: all)"
    )
    
    parser.add_argument(
        "--compare", 
        action="store_true",
        help="Compare results between frameworks"
    )
    
    args = parser.parse_args()
    
    # Setup results directories
    setup_results_dir()
    
    # Determine which frameworks to test
    frameworks = []
    if args.framework == "both":
        frameworks = ["pathrag", "graphrag"]
    else:
        frameworks = [args.framework]
    
    # Run tests
    for framework in frameworks:
        if args.test in ["embedding", "all"]:
            run_embedding_tests(framework)
        
        if args.test in ["path", "all"]:
            run_path_construction_tests(framework)
        
        if args.test in ["query", "all"]:
            run_query_tests(framework)
    
    # Compare results if requested
    if args.compare and (args.framework == "both" or len(frameworks) > 1):
        if args.test in ["embedding", "all"]:
            compare_test_results("embedding")
        
        if args.test in ["path", "all"]:
            compare_test_results("path")
        
        if args.test in ["query", "all"]:
            compare_test_results("query")
    
    logger.info("Testing completed successfully")

if __name__ == "__main__":
    main()
