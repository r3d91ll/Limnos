#!/usr/bin/env python
"""
Run Metadata Transformation Tests.

This script runs the tests for the metadata transformation architecture,
verifying that universal document metadata is correctly transformed into
framework-specific formats.
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
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_tests():
    """Run metadata transformation tests."""
    logger.info("Running metadata transformation tests...")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Add project root to Python path
    sys.path.insert(0, str(project_root))
    
    # Discover and run tests
    test_loader = unittest.TestLoader()
    
    # Run metadata transformation tests
    metadata_test_dir = project_root / "tests" / "metadata"
    metadata_tests = test_loader.discover(str(metadata_test_dir))
    
    # Run integration tests
    integration_test_dir = project_root / "tests" / "integration"
    integration_tests = test_loader.discover(str(integration_test_dir))
    
    # Combine test suites
    all_tests = unittest.TestSuite()
    all_tests.addTests(metadata_tests)
    all_tests.addTests(integration_tests)
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(all_tests)
    
    # Return success/failure
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
