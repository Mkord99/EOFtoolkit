"""
Test Configuration and Setup for EOFtoolkit
This file contains configurations and helper functions for testing
"""

import os
import sys
import tempfile
import numpy as np
import netCDF4 as nc
from pathlib import Path

# Add the project directory to the Python path
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

class TestDataGenerator:
    """Generate test data for EOFtoolkit tests"""
    
    @staticmethod
    def create_test_netcdf_files(directory, num_files=5, nx=10, ny=8):
        """Create a series of test NetCDF files"""
        
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        
        files_created = []
        
        for i in range(num_files):
            # Create filename with date pattern
            filename = f"test_data_{2023+i//12:04d}{(i%12)+1:02d}.nc"
            filepath = os.path.join(directory, filename)
            
            # Create NetCDF file
            with nc.Dataset(filepath, 'w') as ds:
                # Create dimensions
                ds.createDimension('x', nx)
                ds.createDimension('y', ny)
                
                # Create variables
                z = ds.createVariable('z', 'f4', ('y', 'x'))
                x_range = ds.createVariable('x_range', 'f4', (2,))
                y_range = ds.createVariable('y_range', 'f4', (2,))
                dimension = ds.createVariable('dimension', 'f4', (2,))
                spacing = ds.createVariable('spacing', 'f4', (2,))
                
                # Create test data with known patterns
                # Pattern: combination of sine and cosine with random noise
                x = np.linspace(0, 2*np.pi, nx)
                y = np.linspace(0, 2*np.pi, ny)
                X, Y = np.meshgrid(x, y)
                
                # Create data with known modes
                mode1 = np.sin(X) * np.cos(Y)
                mode2 = np.cos(X) * np.sin(Y)
                noise = 0.1 * np.random.randn(ny, nx)
                
                z_data = mode1 + 0.5 * mode2 + noise
                
                # Add some NaN values
                z_data[0, 0] = np.nan
                z_data[-1, -1] = np.nan
                
                # Fill variables
                z[:] = z_data
                x_range[:] = [0, nx]
                y_range[:] = [0, ny]
                dimension[:] = [nx, ny]
                spacing[:] = [1, 1]
                
                # Add attributes
                ds.title = f"Test data for {filename}"
                ds.creation_date = f"2023-{(i%12)+1:02d}-01"
            
            files_created.append(filepath)
        
        return files_created
    
    @staticmethod
    def create_test_matrix_with_known_eofs(n_rows=50, n_cols=100, n_modes=3):
        """Create a test matrix with known EOF patterns"""
        
        # Create time series
        t = np.linspace(0, 4*np.pi, n_rows)
        
        # Create spatial patterns
        x = np.linspace(0, 2*np.pi, n_cols)
        
        # Define known EOFs
        eof1 = np.sin(x)
        eof2 = np.cos(2*x)
        eof3 = np.sin(3*x)
        
        # Define corresponding PCs
        pc1 = np.sin(t)
        pc2 = np.cos(t) * 0.7
        pc3 = np.sin(2*t) * 0.3
        
        # Create matrix
        matrix = np.outer(pc1, eof1) + np.outer(pc2, eof2) + np.outer(pc3, eof3)
        
        # Add some noise
        noise = 0.1 * np.random.randn(n_rows, n_cols)
        matrix += noise
        
        return matrix, [eof1, eof2, eof3], [pc1, pc2, pc3]
    
    @staticmethod
    def create_test_directory_structure():
        """Create test directory structure with sample files"""
        temp_dir = tempfile.mkdtemp(prefix='eof_test_')
        
        # Create directories
        data_dir = os.path.join(temp_dir, 'data')
        results_dir = os.path.join(temp_dir, 'results')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Create test NetCDF files
        files = TestDataGenerator.create_test_netcdf_files(data_dir, num_files=10)
        
        return temp_dir, data_dir, results_dir, files


class TestAssertions:
    """Custom assertion helpers for EOFtoolkit tests"""
    
    @staticmethod
    def assert_eof_properties(eofs, n_modes):
        """Assert EOFs have expected properties"""
        # Check shape
        assert eofs.shape[0] == n_modes
        
        # EOFs should be orthogonal
        if n_modes > 1:
            gram_matrix = eofs @ eofs.T
            # Diagonal should be 1, off-diagonal should be ~0
            np.testing.assert_allclose(np.diag(gram_matrix), np.ones(n_modes), rtol=1e-10)
            # Off-diagonal elements should be small
            off_diag = gram_matrix - np.diag(np.diag(gram_matrix))
            assert np.all(np.abs(off_diag) < 1e-10)
    
    @staticmethod
    def assert_variance_explained(variance, cumulative=None):
        """Assert variance explained properties"""
        # Variance should be positive and decreasing
        assert np.all(variance > 0)
        assert np.all(np.diff(variance) <= 0)
        
        # Total variance should be ~100%
        assert abs(np.sum(variance) - 100.0) < 1e-5
        
        # Check cumulative if provided
        if cumulative is not None:
            assert np.allclose(cumulative, np.cumsum(variance))
            assert cumulative[-1] <= 100.0
    
    @staticmethod
    def assert_reconstruction_accuracy(original, reconstructed, tol=1e-10):
        """Assert reconstruction accuracy"""
        # Should have same shape
        assert original.shape == reconstructed.shape
        
        # Should be close in value
        np.testing.assert_allclose(original, reconstructed, rtol=tol)
        
        # Check error metrics
        mse = np.mean((original - reconstructed)**2)
        assert mse < tol


# Global test configuration
TEST_CONFIG = {
    'random_seed': 42,
    'temp_dir_prefix': 'eof_test_',
    'test_data_size': {
        'small': (10, 20),
        'medium': (50, 100),
        'large': (100, 500)
    },
    'tolerance': {
        'loose': 1e-5,
        'normal': 1e-10,
        'strict': 1e-14
    }
}

# Test decorators
def with_temp_directory(func):
    """Decorator to provide a temporary directory for tests"""
    def wrapper(*args, **kwargs):
        temp_dir = tempfile.mkdtemp(prefix=TEST_CONFIG['temp_dir_prefix'])
        try:
            result = func(*args, temp_dir=temp_dir, **kwargs)
        finally:
            import shutil
            shutil.rmtree(temp_dir)
        return result
    return wrapper

def with_test_data(size='medium'):
    """Decorator to provide test data for tests"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            np.random.seed(TEST_CONFIG['random_seed'])
            
            # Generate test data
            rows, cols = TEST_CONFIG['test_data_size'][size]
            test_matrix = np.random.rand(rows, cols)
            
            return func(*args, test_matrix=test_matrix, **kwargs)
        return wrapper
    return decorator

# Test data validation
def validate_test_data(data, description="test data"):
    """Validate test data structure"""
    required_keys = ['z', 'longitude', 'latitude']
    
    if isinstance(data, dict):
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key '{key}' in {description}")
        
        # Validate shapes
        z_shape = data['z'].shape
        lon_shape = data['longitude'].shape
        lat_shape = data['latitude'].shape
        
        assert lon_shape == lat_shape, f"Longitude and latitude shapes don't match in {description}"
        assert z_shape == lon_shape, f"Data and coordinate shapes don't match in {description}"
    else:
        raise TypeError(f"Expected dict for {description}, got {type(data)}")

# Performance testing helpers
class PerformanceTimer:
    """Context manager for timing test execution"""
    def __init__(self, description="Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        print(f"{self.description} took {self.elapsed:.3f} seconds")