# EOFtoolkit Unit Tests

This directory contains comprehensive unit tests for the EOFtoolkit library.

## Test Structure

```
tests/
├── __init__.py
├── test_svd.py              # Tests for SVD analysis
├── test_reconstruction.py   # Tests for reconstruction functions
├── test_validation.py       # Tests for error validation
├── test_processor.py        # Tests for data processing
├── test_io.py              # Tests for I/O operations
├── test_utils.py           # Tests for utility functions
├── run_tests.py            # Test runner script
├── test_config.py          # Test configurations and helpers
└── conftest.py             # Pytest configuration (optional)
```

## Running Tests

### Using unittest (recommended for this project)

```bash
# Run all tests
python tests/run_tests.py

# Run specific test class
python tests/run_tests.py TestSVD

# Run specific test method
python tests/run_tests.py TestSVD.test_perform_svd_basic

# List all available tests
python tests/run_tests.py --list

# Get help
python tests/run_tests.py --help
```

### Using unittest directly

```bash
# Run all tests
python -m unittest discover tests

# Run with verbose output
python -m unittest discover tests -v

# Run specific test file
python -m unittest tests.test_svd

# Run specific test class
python -m unittest tests.test_svd.TestSVD

# Run specific test method
python -m unittest tests.test_svd.TestSVD.test_perform_svd_basic
```

### Using pytest (optional)

If you prefer pytest, you can install it and use:

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=eoftoolkit --cov-report=html

# Run specific test
pytest tests/test_svd.py::TestSVD::test_perform_svd_basic
```

## Test Categories

### 1. SVD Analysis Tests (test_svd.py)
- Basic SVD functionality
- Variance explained calculations
- EOF orthogonality
- Reconstruction accuracy
- Mode extraction

### 2. Reconstruction Tests (test_reconstruction.py)
- Basic reconstruction
- Incremental reconstruction
- Optimal mode selection
- Error metrics calculation

### 3. Data Processing Tests (test_processor.py)
- Dimension standardization
- Mask creation
- ID matrix generation
- Flattening and centering

### 4. I/O Tests (test_io.py)
- NetCDF file reading
- File sorting by date
- Error handling for invalid files

### 5. Validation Tests (test_validation.py)
- Error metric calculations
- Temporal/spatial error analysis
- R² and variance explained

### 6. Utility Tests (test_utils.py)
- Date extraction from filenames
- File filtering by date range
- General utility functions

## Test Data

Tests use several types of test data:
1. Synthetic NetCDF files with known patterns
2. Random matrices for general testing
3. Matrices with known EOF structures
4. Edge case data (all NaN, single values, etc.)

## Writing New Tests

When adding new tests:

1. Follow the naming convention: `test_*.py` for files, `Test*` for classes
2. Use descriptive test names: `test_function_specific_case`
3. Include setUp/tearDown methods for complex test data
4. Test both normal operation and error conditions
5. Use appropriate assertions (assertEqual, assertRaises, etc.)

Example test structure:

```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.test_data = create_test_data()
    
    def test_normal_operation(self):
        """Test normal operation"""
        result = new_feature(self.test_data)
        self.assertEqual(result, expected_result)
    
    def test_error_handling(self):
        """Test error handling"""
        with self.assertRaises(SpecificError):
            new_feature(invalid_data)
    
    def tearDown(self):
        """Clean up after tests"""
        cleanup_test_data()
```

## Continuous Integration

These tests are designed to be run in CI/CD pipelines. The test runner returns appropriate exit codes:
- 0: All tests passed
- 1: Some tests failed

## Test Coverage

To check test coverage:

```bash
# With unittest
coverage run -m unittest discover tests
coverage report
coverage html  # Generates HTML report

# With pytest
pytest --cov=eoftoolkit --cov-report=html
```

## Troubleshooting

Common issues:

1. **Import errors**: Ensure EOFtoolkit is in your Python path
2. **Missing dependencies**: Install required packages (netCDF4, numpy, etc.)
3. **Temporary file issues**: Make sure /tmp has sufficient space
4. **Random seed issues**: Tests use fixed seeds for reproducibility

## Contact

For issues with tests, please check the main EOFtoolkit documentation or contact the maintainers.