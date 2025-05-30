#!/usr/bin/env python
"""
Comprehensive Test Runner for EOFtoolkit
Runs unit, integration, and validation tests
"""

import unittest
import sys
import os
import time
from pathlib import Path

# Add the project directory to the Python path
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))


class TestSuiteRunner:
    """Run different test suites for EOFtoolkit"""
    
    def __init__(self):
        self.results = {}
    
    def run_unit_tests(self):
        """Run unit tests"""
        print("\n" + "="*70)
        print("RUNNING UNIT TESTS")
        print("="*70)
        
        loader = unittest.TestLoader()
        suite = loader.discover('unit_test', pattern='test_*.py')
        runner = unittest.TextTestRunner(verbosity=2)
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        self.results['unit'] = {
            'passed': result.wasSuccessful(),
            'duration': end_time - start_time,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors)
        }
        
        return result.wasSuccessful()
    
    def run_integration_tests(self):
        """Run integration tests"""
        print("\n" + "="*70)
        print("RUNNING INTEGRATION TESTS")
        print("="*70)
        
        loader = unittest.TestLoader()
        suite = loader.discover('integration_test', pattern='test_*.py')
        runner = unittest.TextTestRunner(verbosity=2)
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        self.results['integration'] = {
            'passed': result.wasSuccessful(),
            'duration': end_time - start_time,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors)
        }
        
        return result.wasSuccessful()
    
    def run_validation_tests(self):
        """Run validation tests"""
        print("\n" + "="*70)
        print("RUNNING VALIDATION TESTS")
        print("="*70)
        
        loader = unittest.TestLoader()
        suite = loader.discover('validation_test', pattern='test_*.py')
        runner = unittest.TextTestRunner(verbosity=2)
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        self.results['validation'] = {
            'passed': result.wasSuccessful(),
            'duration': end_time - start_time,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors)
        }
        
        return result.wasSuccessful()
    
    def run_all_tests(self):
        """Run all test suites"""
        print("Starting EOFtoolkit Comprehensive Test Suite")
        print("="*70)
        
        all_passed = True
        
        # Run each test suite
        all_passed &= self.run_unit_tests()
        all_passed &= self.run_integration_tests()
        all_passed &= self.run_validation_tests()
        
        # Print summary
        self.print_summary()
        
        return all_passed
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_duration = 0
        
        for test_type, result in self.results.items():
            print(f"\n{test_type.upper()} TESTS:")
            print(f"  Status: {'PASSED' if result['passed'] else 'FAILED'}")
            print(f"  Tests run: {result['tests_run']}")
            print(f"  Failures: {result['failures']}")
            print(f"  Errors: {result['errors']}")
            print(f"  Duration: {result['duration']:.2f} seconds")
            
            total_tests += result['tests_run']
            total_failures += result['failures']
            total_errors += result['errors']
            total_duration += result['duration']
        
        print(f"\nTOTAL:")
        print(f"  Tests run: {total_tests}")
        print(f"  Failures: {total_failures}")
        print(f"  Errors: {total_errors}")
        print(f"  Duration: {total_duration:.2f} seconds")
        
        success_rate = ((total_tests - total_failures - total_errors) / total_tests) * 100
        print(f"  Success rate: {success_rate:.1f}%")
        
        print("="*70)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run EOFtoolkit tests')
    parser.add_argument('--type', choices=['unit', 'integration', 'validation', 'all'],
                        default='all', help='Type of tests to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    runner = TestSuiteRunner()
    
    if args.type == 'unit':
        success = runner.run_unit_tests()
    elif args.type == 'integration':
        success = runner.run_integration_tests()
    elif args.type == 'validation':
        success = runner.run_validation_tests()
    else:
        success = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()