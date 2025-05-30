"""
Integration Tests for EOFProcessor class
Tests how different EOFProcessor methods work together
"""

import unittest
import tempfile
import shutil
import numpy as np
import netCDF4 as nc
from pathlib import Path
import sys

# Add project directory to path
project_dir = Path(__file__).parents[2]
sys.path.insert(0, str(project_dir))

from eoftoolkit.core.processor import EOFProcessor


class TestEOFProcessorIntegration(unittest.TestCase):
    """Integration tests for EOFProcessor class methods"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp(prefix='eof_processor_test_')
        self.data_dir = Path(self.test_dir) / 'data'
        self.data_dir.mkdir()
        
        # Create test data with known properties
        self.create_synthetic_netcdf_files()
        
        # Initialize processor
        self.processor = EOFProcessor(verbose=False)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def create_synthetic_netcdf_files(self):
        """Create NetCDF files with known EOF patterns"""
        # Create 10 files with known temporal and spatial patterns
        for t in range(10):
            filename = f"synthetic_{t:03d}.nc"
            filepath = self.data_dir / filename
            
            with nc.Dataset(str(filepath), 'w') as ds:
                nx, ny = 30, 25
                ds.createDimension('x', nx)
                ds.createDimension('y', ny)
                ds.createDimension('two', 2)
                
                # Create variables
                z = ds.createVariable('z', 'f4', ('y', 'x'))
                x_range = ds.createVariable('x_range', 'f4', ('two',))
                y_range = ds.createVariable('y_range', 'f4', ('two',))
                dimension = ds.createVariable('dimension', 'f4', ('two',))
                spacing = ds.createVariable('spacing', 'f4', ('two',))
                
                # Create spatial patterns
                x = np.linspace(0, 4*np.pi, nx)
                y = np.linspace(0, 4*np.pi, ny)
                X, Y = np.meshgrid(x, y)
                
                # EOF 1: Dipole pattern
                eof1 = np.sin(X/2) * np.cos(Y/2)
                
                # EOF 2: Quadrupole pattern
                eof2 = np.sin(X) * np.sin(Y)
                
                # EOF 3: Complex pattern
                eof3 = np.sin(2*X) * np.cos(2*Y) + np.cos(3*X) * np.sin(Y)
                
                # Temporal variations (known PCs)
                time = t / 10 * 2 * np.pi
                pc1 = np.sin(time)
                pc2 = np.cos(time) * 0.7
                pc3 = np.sin(3*time) * 0.3
                
                # Combine with small noise
                noise = 0.05 * np.random.randn(ny, nx)
                z_data = pc1 * eof1 + pc2 * eof2 + pc3 * eof3 + noise
                
                # Fill variables
                z[:] = z_data
                x_range[:] = [0, nx]
                y_range[:] = [0, ny]
                dimension[:] = [nx, ny]
                spacing[:] = [1, 1]
    
    def test_processor_state_transitions(self):
        """Test state transitions of EOFProcessor"""
        # Initial state
        self.assertIsNone(self.processor.super_matrix)
        self.assertIsNone(self.processor.svd_results)
        self.assertIsNone(self.processor.reconstruction_results)
        
        # After processing directory
        self.processor.process_directory(str(self.data_dir))
        self.assertIsNotNone(self.processor.super_matrix)
        self.assertIsNone(self.processor.svd_results)
        
        # After SVD
        self.processor.perform_svd(num_modes=3)
        self.assertIsNotNone(self.processor.svd_results)
        self.assertIsNone(self.processor.reconstruction_results)
        
        # After reconstruction
        self.processor.reconstruct()
        self.assertIsNotNone(self.processor.reconstruction_results)
        
        # Test reset
        self.processor.reset()
        self.assertIsNone(self.processor.super_matrix)
        self.assertIsNone(self.processor.svd_results)
        self.assertIsNone(self.processor.reconstruction_results)
    
    def test_mode_extraction_and_analysis(self):
        """Test extraction and analysis of individual modes"""
        # Process data
        self.processor.process_directory(str(self.data_dir))
        self.processor.perform_svd(num_modes=3)
        
        # Test EOF extraction
        eof1 = self.processor.get_eof(1, reshape=True)
        eof2 = self.processor.get_eof(2, reshape=True)
        
        # EOFs should be orthogonal
        eof1_flat = eof1[~np.isnan(eof1)].flatten()
        eof2_flat = eof2[~np.isnan(eof2)].flatten()
        dot_product = np.dot(eof1_flat, eof2_flat)
        self.assertAlmostEqual(dot_product, 0, places=10)
        
        # Test PC extraction
        pc1 = self.processor.get_pc(1)
        pc2 = self.processor.get_pc(2)
        
        # PCs should be orthogonal
        dot_product = np.dot(pc1, pc2)
        self.assertAlmostEqual(dot_product, 0, places=10)
        
        # Check PC shapes
        self.assertEqual(pc1.shape, (10,))  # 10 time steps
        self.assertEqual(pc2.shape, (10,))
    
    def test_incremental_reconstruction(self):
        """Test incremental reconstruction with different numbers of modes"""
        # Process data and perform SVD
        self.processor.process_directory(str(self.data_dir))
        self.processor.perform_svd(num_modes=4)
        self.processor.reconstruct(max_modes=4)
        
        # Test reconstructions with different mode counts
        errors = []
        for n_modes in range(1, 5):
            recon = self.processor.get_reconstruction(mode_count=n_modes, timestamp_index=0)
            original = self.processor.get_original_data(timestamp_index=0)
            
            # Calculate error
            valid_mask = ~(np.isnan(recon) | np.isnan(original))
            error = np.sqrt(np.mean((recon[valid_mask] - original[valid_mask])**2))
            errors.append(error)
        
        # Error should decrease with more modes
        for i in range(1, len(errors)):
            self.assertLess(errors[i], errors[i-1])
    
    def test_date_handling(self):
        """Test date extraction and handling"""
        # Process with specific date pattern
        self.processor.process_directory(str(self.data_dir), date_pattern=r'(\d{3})', date_format=None)
        
        # Get dates as strings
        dates_str = self.processor.get_dates()
        self.assertEqual(len(dates_str), 10)
        self.assertTrue(all(isinstance(d, str) for d in dates_str))
        
        # Dates should be sorted
        self.assertEqual(dates_str, sorted(dates_str))
    
    def test_visualization_methods_work_together(self):
        """Test that different visualization methods work together"""
        # Process data
        self.processor.process_directory(str(self.data_dir))
        self.processor.perform_svd(num_modes=3)
        self.processor.reconstruct()
        
        # Store visualization figures to ensure they don't interfere
        figures = []
        
        # EOF visualization
        for mode in [1, 2, 3]:
            fig, ax = self.processor.visualize_eof(mode)
            figures.append((fig, ax))
            self.assertIsNotNone(fig)
        
        # PC visualization
        for mode in [1, 2, 3]:
            fig, ax = self.processor.visualize_pc(mode, dates=self.processor.get_dates())
            figures.append((fig, ax))
            self.assertIsNotNone(fig)
        
        # Reconstruction visualization
        fig, ax = self.processor.visualize_reconstruction(mode_count=2, timestamp_index=0)
        figures.append((fig, ax))
        self.assertIsNotNone(fig)
        
        # Error visualization
        fig, ax = self.processor.visualize_reconstruction_error()
        figures.append((fig, ax))
        self.assertIsNotNone(fig)
        
        # Comparison visualization
        fig, axes = self.processor.visualize_comparison(mode_count=2, timestamp_index=0)
        figures.append((fig, axes))
        self.assertIsNotNone(fig)


if __name__ == '__main__':
    unittest.main()