#!/usr/bin/env python
"""
Test Runner for EOFtoolkit
This script runs all unit tests and generates a test report
"""

import unittest
import sys
import os
from datetime import datetime

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """Run all unit tests and generate a report"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='test_*.py')
    
    # Create test runner with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests
    print("=" * 70)
    print(f"EOFtoolkit Unit Test Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    # Calculate success rate
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    # Return exit code (0 for success, 1 for failures)
    return 0 if result.wasSuccessful() else 1

def run_specific_test(test_name):
    """Run a specific test class or method"""
    loader = unittest.TestLoader()
    
    try:
        # Try to load the specific test
        suite = loader.loadTestsFromName(test_name)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return 0 if result.wasSuccessful() else 1
    except Exception as e:
        print(f"Error loading test '{test_name}': {e}")
        return 1

def list_available_tests():
    """List all available tests"""
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    print("Available test classes:")
    print("-" * 50)
    
    for test_group in suite:
        for test_case in test_group:
            if hasattr(test_case, '_testMethodName'):
                test_class = test_case.__class__.__name__
                test_method = test_case._testMethodName
                print(f"{test_class}.{test_method}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_available_tests()
        elif sys.argv[1] == "--help":
            print("""
EOFtoolkit Test Runner

Usage:
  python run_tests.py              # Run all tests
  python run_tests.py --list       # List available tests
  python run_tests.py <test_name>  # Run specific test
  
Examples:
  python run_tests.py TestSVD                     # Run all SVD tests
  python run_tests.py TestSVD.test_perform_svd    # Run specific test method
            """)
        else:
            # Run specific test
            sys.exit(run_specific_test(sys.argv[1]))
    else:
        # Run all tests
        sys.exit(run_all_tests())