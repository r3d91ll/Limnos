#!/usr/bin/env python3
"""
Run all universal component tests for the Limnos framework.

This script runs the test suites for all universal components:
1. Universal Document Collector
2. Metadata Provider
3. Document Processor

Usage:
    python run_universal_tests.py
"""

import os
import sys
import unittest
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
logger.info(f"Added {project_root} to Python path")


def run_tests():
    """Run all universal component tests."""
    logger.info("Starting universal component tests for Limnos")
    
    # Discover and run all tests in the universal directory
    test_dir = Path(__file__).parent.parent / "tests" / "universal"
    
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return False
    
    logger.info(f"Discovering tests in {test_dir}")
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern="test_*.py")
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Log results
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    
    # Return True if all tests passed
    return len(result.errors) == 0 and len(result.failures) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
