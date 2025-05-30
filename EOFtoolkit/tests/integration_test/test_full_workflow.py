"""
Integration Tests for EOFtoolkit - Full Workflow
Tests the complete pipeline from data reading to visualization
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
from eoftoolkit.io.reader import read_netcdf
from eoftoolkit.io.writer import save_results


class TestFullWorkflow(unittest.TestCase):
    """Integration tests for the complete EOF analysis workflow"""
    
    def setUp(self):
        """Set up test environment with sample data"""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp(prefix='eof_integration_test_')
        self.data_dir = Path(self.test_dir) / 'data'
        self.output_dir = Path(self.test_dir) / 'output'
        self.data_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create test NetCDF files
        self.create_test_netcdf_files()
        
        # Initialize processor
        self.processor = EOFProcessor(verbose=False)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def create_test_netcdf_files(self, num_files=5):
        """Create test NetCDF files with known patterns"""
        for i in range(num_files):
            filename = f"test_data_2023{i+1:02d}.nc"
            filepath = self.data_dir / filename
            
            with nc.Dataset(str(filepath), 'w') as ds:
                # Create dimensions
                nx, ny = 20, 15
                ds.createDimension('x', nx)
                ds.createDimension('y', ny)
                ds.createDimension('two', 2)
                
                # Create variables
                z = ds.createVariable('z', 'f4', ('y', 'x'))
                x_range = ds.createVariable('x_range', 'f4', ('two',))
                y_range = ds.createVariable('y_range', 'f4', ('two',))
                dimension = ds.createVariable('dimension', 'f4', ('two',))
                spacing = ds.createVariable('spacing', 'f4', ('two',))
                
                # Create test data with known EOF patterns
                x = np.linspace(0, 2*np.pi, nx)
                y = np.linspace(0, 2*np.pi, ny)
                X, Y = np.meshgrid(x, y)
                
                # Create two dominant modes
                mode1 = np.sin(X) * np.cos(Y)
                mode2 = np.cos(2*X) * np.sin(2*Y)
                
                # Add time-varying amplitude with some noise
                t = i / num_files * 2 * np.pi
                amplitude1 = np.sin(t)
                amplitude2 = 0.5 * np.cos(t)
                noise = 0.1 * np.random.randn(ny, nx)
                
                z_data = amplitude1 * mode1 + amplitude2 * mode2 + noise
                
                # Fill variables
                z[:] = z_data
                x_range[:] = [0, nx]
                y_range[:] = [0, ny]
                dimension[:] = [nx, ny]
                spacing[:] = [1, 1]
    
    def test_complete_workflow(self):
        """Test the complete EOF analysis workflow"""
        # Step 1: Process directory
        results = self.processor.process_directory(
            str(self.data_dir),
            file_extension='.nc',
            date_pattern=r'(\d{6})',
            date_format='%Y%m'
        )
        
        # Verify processing results
        self.assertIn('super_matrix', results)
        self.assertIn('id_matrix', results)
        self.assertIn('super_mask', results)
        self.assertEqual(results['super_matrix'].shape[0], 5)  # 5 time steps
        
        # Step 2: Perform SVD
        svd_results = self.processor.perform_svd(num_modes=3)
        
        # Verify SVD results
        self.assertIn('eofs', svd_results)
        self.assertIn('pcs', svd_results)
        self.assertEqual(svd_results['eofs'].shape[0], 3)
        
        # Check that first two modes capture most variance
        total_variance = sum(svd_results['explained_variance'][:2])
        self.assertGreater(total_variance, 80.0)  # Should explain >80% of variance
        
        # Step 3: Perform reconstruction
        recon_results = self.processor.reconstruct(max_modes=3)
        
        # Verify reconstruction results
        self.assertIn('optimal_reconstruction', recon_results)
        self.assertIn('optimal_mode_count', recon_results)
        self.assertIn('error_metrics', recon_results)
        
        # Check reconstruction accuracy
        optimal_error = recon_results['error_metrics'][recon_results['optimal_mode_count']]['rmse']
        self.assertLess(optimal_error, 0.5)  # RMSE should be reasonably small
        
        # Step 4: Get and verify reconstructed data
        reconstructed = self.processor.get_reconstruction(mode_count=2, timestamp_index=0)
        original = self.processor.get_original_data(timestamp_index=0)
        
        # Shapes should match
        self.assertEqual(reconstructed.shape, original.shape)
        
        # Step 5: Test visualization methods (don't save, just check they run)
        # Test EOF visualization
        fig1, ax1 = self.processor.visualize_eof(1)
        self.assertIsNotNone(fig1)
        self.assertIsNotNone(ax1)
        
        # Test PC visualization
        fig2, ax2 = self.processor.visualize_pc(1)
        self.assertIsNotNone(fig2)
        self.assertIsNotNone(ax2)
        
        # Test reconstruction visualization
        fig3, ax3 = self.processor.visualize_reconstruction(mode_count=2, timestamp_index=0)
        self.assertIsNotNone(fig3)
        self.assertIsNotNone(ax3)
        
        # Test error visualization
        fig4, ax4 = self.processor.visualize_reconstruction_error()
        self.assertIsNotNone(fig4)
        self.assertIsNotNone(ax4)
        
        # Test comparison visualization
        fig5, axes5 = self.processor.visualize_comparison(mode_count=2, timestamp_index=0)
        self.assertIsNotNone(fig5)
        self.assertIsNotNone(axes5)
        
        # Step 6: Save results
        saved_files = self.processor.save_results(str(self.output_dir), prefix='integration_test')
        
        # Verify files were saved
        self.assertIn('eofs', saved_files)
        self.assertIn('pcs', saved_files)
        self.assertTrue(Path(saved_files['eofs']).exists())
        self.assertTrue(Path(saved_files['pcs']).exists())
    
    def test_date_filtering(self):
        """Test workflow with date filtering"""
        # Test with date range
        results = self.processor.process_directory(
            str(self.data_dir),
            file_extension='.nc',
            date_pattern=r'(\d{6})',
            date_format='%Y%m',
            start_date='2023-02-01',
            end_date='2023-04-30'
        )
        
        # Should only process files 2-4
        self.assertEqual(results['super_matrix'].shape[0], 3)
    
    def test_projection_configuration(self):
        """Test different map projections"""
        # Process data first
        self.processor.process_directory(str(self.data_dir))
        self.processor.perform_svd(num_modes=2)
    
        # Test with different projections
        projections = {
            'merc': {},
            'lcc': {
                'lat_1': 30,
                'lat_2': 60,
                'lat_0': 45,
                'lon_0': -90
            },
            'stere': {
                'lat_0': 45,
                'lon_0': -90
            }
        }
    
        for proj, params in projections.items():
            self.processor.configure_projection(projection=proj, projection_params=params)
        
        # Test visualization with different projections
            fig, ax = self.processor.visualize_eof(1, **params)  # Pass params directly
            self.assertIsNotNone(fig)
            self.assertIsNotNone(ax)
    
    def test_error_handling_workflow(self):
        """Test error handling in the workflow"""
        # Test with invalid directory
        with self.assertRaises(Exception):
            self.processor.process_directory('/nonexistent/directory')
        
        # Test SVD before processing
        with self.assertRaises(Exception):
            self.processor.perform_svd()
        
        # Test reconstruction before SVD
        self.processor.process_directory(str(self.data_dir))
        with self.assertRaises(Exception):
            self.processor.reconstruct()


if __name__ == '__main__':
    unittest.main()