import sys
import os
from pathlib import Path

# Add EOFtoolkit directory to Python path
project_dir = Path(__file__).parents[2]  # Go up 2 levels to EOFtoolkit directory
sys.path.insert(0, str(project_dir))

# Now import the rest
import unittest
import numpy as np
import netCDF4 as nc
import tempfile
import os
from datetime import datetime
import shutil

# Import EOFtoolkit modules
from eoftoolkit.analysis.svd import perform_svd, extract_modes
from eoftoolkit.analysis.reconstruction import reconstruct_from_modes, add_means_back
from eoftoolkit.analysis.validation import calculate_error_metrics, calculate_temporal_error_metrics, calculate_spatial_error_metrics
from eoftoolkit.processor.dimensions import standardize_dimensions
from eoftoolkit.processor.masking import create_binary_mask, create_super_mask
from eoftoolkit.processor.identification import create_id_matrix, get_id_coordinates
from eoftoolkit.processor.flattener import flatten_matrices, center_matrices
from eoftoolkit.processor.reshaper import reshape_to_spatial_grid, reshape_all_to_spatial_grid
from eoftoolkit.processor.stacker import create_super_matrix
from eoftoolkit.io.reader import read_netcdf
from eoftoolkit.io.sorter import sort_files_by_date
from eoftoolkit.core.utils import extract_date_from_filename, filter_files_by_date_range
from eoftoolkit.core.exceptions import EOFToolkitError, FileReadError, DimensionError, SVDError, ReconstructionError

class TestSVD(unittest.TestCase):
    """Test SVD analysis functions"""
    
    def setUp(self):
        """Create test data for SVD tests"""
        # Create simple test matrix with known properties
        self.simple_matrix = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0]
        ])
        
        # Create random matrix for general testing
        np.random.seed(42)
        self.random_matrix = np.random.rand(10, 20)
        
        # Create matrix with known singular values
        self.orthogonal_matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
    
    def test_perform_svd_basic(self):
        """Test basic SVD functionality"""
        results = perform_svd(self.simple_matrix)
        
        # Check all required keys are present
        self.assertIn('eofs', results)
        self.assertIn('pcs', results)
        self.assertIn('singular_values', results)
        self.assertIn('explained_variance', results)
        self.assertIn('cumulative_variance', results)
        
        # Check dimensions
        n_modes = len(results['singular_values'])
        self.assertEqual(results['eofs'].shape[0], n_modes)
        self.assertEqual(results['pcs'].shape[1], n_modes)
    
    def test_perform_svd_num_modes(self):
        """Test SVD with specified number of modes"""
        num_modes = 2
        results = perform_svd(self.random_matrix, num_modes=num_modes)
        
        self.assertEqual(results['eofs'].shape[0], num_modes)
        self.assertEqual(results['pcs'].shape[1], num_modes)
        self.assertEqual(len(results['singular_values']), num_modes)
    
    def test_perform_svd_variance_explained(self):
        """Test variance explained calculation"""
        results = perform_svd(self.random_matrix)
        
        # Variance should sum to 100%
        self.assertAlmostEqual(np.sum(results['explained_variance']), 100.0, places=5)
        
        # Cumulative variance should be sorted
        self.assertTrue(np.all(np.diff(results['cumulative_variance']) >= 0))
        
        # First component should have highest variance
        self.assertTrue(results['explained_variance'][0] >= results['explained_variance'][1])
    
    def test_perform_svd_reconstruction(self):
        """Test reconstruction from SVD results"""
        results = perform_svd(self.random_matrix)
        
        # Reconstruct using all modes
        eofs = results['eofs']
        pcs = results['pcs']
        reconstructed = pcs @ eofs
        
        # Should match original matrix
        np.testing.assert_allclose(reconstructed, self.random_matrix, rtol=1e-10)
    
    def test_perform_svd_corresponding_surfaces(self):
        """Test computation of corresponding surfaces"""
        results = perform_svd(self.simple_matrix, compute_surfaces=True)
        
        self.assertIn('corresponding_surfaces', results)
        
        # Check first surface
        mode_1_surface = results['corresponding_surfaces']['mode_1']
        expected_shape = (self.simple_matrix.shape[0], self.simple_matrix.shape[1])
        self.assertEqual(mode_1_surface.shape, expected_shape)
    
    def test_extract_modes(self):
        """Test extracting specific modes"""
        results = perform_svd(self.random_matrix)
        modes_to_extract = [1, 3]
        
        extracted = extract_modes(results, modes_to_extract)
        
        self.assertEqual(extracted['eofs'].shape[0], len(modes_to_extract))
        self.assertEqual(extracted['pcs'].shape[1], len(modes_to_extract))
        
        # Check extracted values match original
        np.testing.assert_array_equal(extracted['eofs'][0], results['eofs'][0])
        np.testing.assert_array_equal(extracted['eofs'][1], results['eofs'][2])
    
    def test_perform_svd_with_zero_matrix(self):
        """Test SVD with all-zero matrix"""
        zero_matrix = np.zeros((5, 5))
        
        with self.assertRaises(SVDError):
            results = perform_svd(zero_matrix)


class TestReconstruction(unittest.TestCase):
    """Test reconstruction functions"""
    
    def setUp(self):
        """Create test data for reconstruction tests"""
        np.random.seed(42)
        
        # Create simple test data
        self.original_data = np.random.rand(10, 20)
        
        # Perform SVD for reconstruction tests
        self.svd_results = perform_svd(self.original_data)
        self.svd_results['super_matrix'] = self.original_data
    
    def test_reconstruct_from_modes_basic(self):
        """Test basic reconstruction functionality"""
        recon_results = reconstruct_from_modes(self.svd_results)
        
        # Check required keys
        self.assertIn('reconstructions', recon_results)
        self.assertIn('optimal_reconstruction', recon_results)
        self.assertIn('optimal_mode_count', recon_results)
        self.assertIn('error_metrics', recon_results)
        
        # Check reconstruction shapes
        n_modes = self.svd_results['eofs'].shape[0]
        for i in range(1, n_modes + 1):
            self.assertIn(i, recon_results['reconstructions'])
            self.assertEqual(recon_results['reconstructions'][i].shape, self.original_data.shape)
    
    def test_reconstruct_from_modes_accuracy(self):
        """Test reconstruction accuracy"""
        recon_results = reconstruct_from_modes(self.svd_results)
        
        # Full reconstruction should match original
        full_reconstruction = recon_results['reconstructions'][len(self.svd_results['singular_values'])]
        np.testing.assert_allclose(full_reconstruction, self.original_data, rtol=1e-10)
    
    def test_reconstruct_from_modes_max_modes(self):
        """Test reconstruction with limited modes"""
        max_modes = 3
        recon_results = reconstruct_from_modes(self.svd_results, max_modes=max_modes)
        
        # Should only have reconstructions up to max_modes
        self.assertEqual(len(recon_results['reconstructions']), max_modes)
        self.assertIn(max_modes, recon_results['reconstructions'])
        self.assertNotIn(max_modes + 1, recon_results['reconstructions'])
    
    def test_reconstruct_from_modes_error_metrics(self):
        """Test error metrics in reconstruction"""
        recon_results = reconstruct_from_modes(self.svd_results)
        
        # Error should decrease with more modes
        error_values = []
        for i in range(1, len(recon_results['error_metrics']) + 1):
            error_values.append(recon_results['error_metrics'][i]['rmse'])
        
        # RMSE should generally decrease
        self.assertTrue(error_values[-1] < error_values[0])
    
    def test_add_means_back(self):
        """Test adding means back to reconstructions"""
        # Create test data with known means
        data_dict = {
            'data1': np.array([[0, 0, 0], [0, 0, 0]]),
            'data2': np.array([[1, 1, 1], [1, 1, 1]])
        }
        
        mean_dict = {
            'data1': np.array([1, 2, 3]),
            'data2': np.array([4, 5, 6])
        }
        
        reconstructions = {
            1: np.vstack([data_dict['data1'][0], data_dict['data2'][0]])
        }
        
        result = add_means_back(reconstructions, mean_dict, ['data1', 'data2'])
        
        # Check means were added correctly
        np.testing.assert_array_equal(result[1][0], [1, 2, 3])
        np.testing.assert_array_equal(result[1][1], [5, 6, 7])


class TestValidation(unittest.TestCase):
    """Test validation functions"""
    
    def setUp(self):
        """Create test data for validation tests"""
        np.random.seed(42)
        self.original = np.random.rand(10, 20)
        self.identical = self.original.copy()
        self.different = self.original + 0.1
    
    def test_calculate_error_metrics_perfect_match(self):
        """Test error metrics with identical data"""
        metrics = calculate_error_metrics(self.original, self.identical)
        
        # Perfect match should have zero error
        self.assertAlmostEqual(metrics['mse'], 0.0, places=10)
        self.assertAlmostEqual(metrics['rmse'], 0.0, places=10)
        self.assertAlmostEqual(metrics['mae'], 0.0, places=10)
        self.assertAlmostEqual(metrics['max_error'], 0.0, places=10)
        self.assertAlmostEqual(metrics['r2'], 1.0, places=10)
    
    def test_calculate_error_metrics_known_values(self):
        """Test error metrics with known differences"""
        original = np.ones((2, 2))
        reconstruction = np.zeros((2, 2))
        
        metrics = calculate_error_metrics(original, reconstruction)
        
        # Known values
        self.assertAlmostEqual(metrics['mse'], 1.0)
        self.assertAlmostEqual(metrics['rmse'], 1.0)
        self.assertAlmostEqual(metrics['mae'], 1.0)
        self.assertAlmostEqual(metrics['max_error'], 1.0)
    
    def test_calculate_temporal_error_metrics(self):
        """Test temporal error metrics"""
        metrics = calculate_temporal_error_metrics(self.original, self.different)
        
        # Check shape
        self.assertEqual(metrics['rmse'].shape, (self.original.shape[0],))
        self.assertEqual(metrics['mae'].shape, (self.original.shape[0],))
        
        # All values should be positive
        self.assertTrue(np.all(metrics['rmse'] > 0))
        self.assertTrue(np.all(metrics['mae'] > 0))
    
    def test_calculate_spatial_error_metrics(self):
        """Test spatial error metrics"""
        metrics = calculate_spatial_error_metrics(self.original, self.different)
        
        # Check shape
        self.assertEqual(metrics['rmse'].shape, (self.original.shape[1],))
        self.assertEqual(metrics['mae'].shape, (self.original.shape[1],))
        
        # All values should be positive
        self.assertTrue(np.all(metrics['rmse'] > 0))
        self.assertTrue(np.all(metrics['mae'] > 0))


class TestProcessor(unittest.TestCase):
    """Test processor functions"""
    
    def setUp(self):
        """Create test data for processor tests"""
        self.matrices = {
            'mat1': np.array([[1, 2], [3, 4]]),
            'mat2': np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]]),
            'mat3': np.array([[14, 15, 16], [17, 18, 19]])
        }
        
        self.nan_matrix = np.array([[1, 2, np.nan], [np.nan, 5, 6], [7, np.nan, 9]])
    
    def test_standardize_dimensions_basic(self):
        """Test basic dimension standardization"""
        result, target_dims = standardize_dimensions(self.matrices)
        
        # All matrices should have same dimensions
        for mat in result.values():
            self.assertEqual(mat.shape, target_dims)
        
        # Target dims should be max of all matrices
        self.assertEqual(target_dims, (3, 3))
        
        # Check padding with NaN
        self.assertTrue(np.isnan(result['mat1'][2, 2]))
    
    def test_standardize_dimensions_with_target(self):
        """Test dimension standardization with specified target"""
        target_dims = (5, 4)
        result, actual_dims = standardize_dimensions(self.matrices, target_dims=target_dims)
        
        self.assertEqual(actual_dims, target_dims)
        
        # All matrices should have target dimensions
        for mat in result.values():
            self.assertEqual(mat.shape, target_dims)
    
    def test_create_binary_mask_with_nans(self):
        """Test binary mask creation with NaN values"""
        mask = create_binary_mask(self.nan_matrix)
        
        expected = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        np.testing.assert_array_equal(mask, expected)
    
    def test_create_binary_mask_with_masked_array(self):
        """Test binary mask creation with masked arrays"""
        # Create a test matrix with known values
        test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
        # Mask values greater than 6
        masked_array = np.ma.masked_where(test_matrix > 6, test_matrix)
        mask = create_binary_mask(masked_array)
    
        # Should mark values > 6 as 0
        self.assertEqual(mask[2, 0], 0)  # Value 7 should be marked
        self.assertEqual(mask[2, 1], 0)  # Value 8 should be marked  
        self.assertEqual(mask[2, 2], 0)  # Value 9 should be marked
    
        # Values <= 6 should be 1
        self.assertEqual(mask[1, 1], 1)  # Value 5 should not be marked
        self.assertEqual(mask[0, 0], 1)  # Value 1 should not be marked

    def test_create_super_mask_all_required(self):
        """Test super mask creation with all masks required"""
        masks = {
            'mask1': np.array([[1, 1, 0], [0, 1, 1]]),
            'mask2': np.array([[1, 0, 0], [1, 1, 1]]),
            'mask3': np.array([[1, 1, 1], [0, 0, 1]])
        }
        
        super_mask = create_super_mask(masks)
        expected = np.array([[1, 0, 0], [0, 0, 1]])
        np.testing.assert_array_equal(super_mask, expected)
    
    def test_create_super_mask_with_threshold(self):
        """Test super mask creation with threshold"""
        masks = {
            'mask1': np.array([[1, 1, 0], [0, 1, 1]]),
            'mask2': np.array([[1, 0, 0], [1, 1, 1]]),
            'mask3': np.array([[1, 1, 1], [0, 0, 1]])
        }
        
        super_mask = create_super_mask(masks, threshold=2)
        expected = np.array([[1, 1, 0], [0, 1, 1]])
        np.testing.assert_array_equal(super_mask, expected)


class TestIdentification(unittest.TestCase):
    """Test ID matrix functions"""
    
    def setUp(self):
        """Create test data for ID tests"""
        self.super_mask = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    
    def test_create_id_matrix_default(self):
        """Test ID matrix creation with default settings"""
        id_matrix = create_id_matrix(self.super_mask)
        
        # Check shape
        self.assertEqual(id_matrix.shape, self.super_mask.shape)
        
        # Check ID format (should be '0101' style)
        self.assertEqual(id_matrix[0, 0], '0101')
        self.assertEqual(id_matrix[0, 1], '0102')
        self.assertEqual(id_matrix[0, 2], '')  # Masked cell
        self.assertEqual(id_matrix[1, 1], '0202')
    
    def test_create_id_matrix_custom_format(self):
        """Test ID matrix creation with custom format"""
        id_matrix = create_id_matrix(self.super_mask, base=0, format_string='{:d}_{:d}')
        
        # Check custom format
        self.assertEqual(id_matrix[0, 0], '0_0')
        self.assertEqual(id_matrix[1, 1], '1_1')
    
    def test_get_id_coordinates(self):
        """Test getting coordinates for IDs"""
        id_matrix = create_id_matrix(self.super_mask)
        lons = np.array([[10, 11, 12], [10, 11, 12], [10, 11, 12]])
        lats = np.array([[20, 20, 20], [21, 21, 21], [22, 22, 22]])
        
        coordinates = get_id_coordinates(id_matrix, lons, lats)
        
        # Check number of coordinates
        valid_cells = np.sum(self.super_mask)
        self.assertEqual(len(coordinates), valid_cells)
        
        # Check first coordinate
        self.assertEqual(coordinates[0]['id'], '0101')
        self.assertEqual(coordinates[0]['longitude'], 10)
        self.assertEqual(coordinates[0]['latitude'], 20)


class TestFlattener(unittest.TestCase):
    """Test flattening functions"""
    
    def setUp(self):
        """Create test data for flattening tests"""
        self.super_mask = np.array([[1, 1, 0], [0, 1, 1]])
        self.id_matrix = create_id_matrix(self.super_mask)
        
        self.matrices = {
            'mat1': np.array([[1, 2, 3], [4, 5, 6]]),
            'mat2': np.array([[7, 8, 9], [10, 11, 12]])
        }
        
        self.masked_matrix = np.ma.masked_array(
            [[1, 2, 3], [4, 5, 6]],
            mask=[[0, 0, 1], [1, 0, 0]]
        )
    
    def test_flatten_matrices_basic(self):
        """Test basic matrix flattening"""
        flattened, fl_id_matrix = flatten_matrices(self.matrices, self.id_matrix, self.super_mask)
        
        # Check shapes
        n_valid_cells = np.sum(self.super_mask)
        for key, matrix in flattened.items():
            self.assertEqual(matrix.shape, (1, n_valid_cells))
        
        # Check ID matrix shape
        self.assertEqual(fl_id_matrix.shape, (1, n_valid_cells))
    
    def test_flatten_matrices_with_masked_array(self):
        """Test flattening with masked arrays"""
        matrices = {'masked': self.masked_matrix}
        flattened, _ = flatten_matrices(matrices, self.id_matrix, self.super_mask)
        
        # Should extract unmasked values correctly
        self.assertEqual(flattened['masked'].shape, (1, 4))  # 4 valid cells
    
    def test_center_matrices_basic(self):
        """Test matrix centering"""
        matrices = {
            'mat1': np.array([[1, 2, 3, 4]]),
            'mat2': np.array([[5, 6, 7, 8]])
        }
        
        centered, means = center_matrices(matrices, axis=1, return_means=True)
        
        # Check means are correct
        self.assertAlmostEqual(means['mat1'][0, 0], 2.5)
        self.assertAlmostEqual(means['mat2'][0, 0], 6.5)
        
        # Check centering is correct
        np.testing.assert_allclose(centered['mat1'], [[-1.5, -0.5, 0.5, 1.5]])
        np.testing.assert_allclose(centered['mat2'], [[-1.5, -0.5, 0.5, 1.5]])


class TestReshaper(unittest.TestCase):
    """Test reshaping functions"""
    
    def setUp(self):
        """Create test data for reshaping tests"""
        self.super_mask = np.array([[1, 1, 0], [0, 1, 1]])
        self.id_matrix = create_id_matrix(self.super_mask)
        self.flattened_data = np.array([10, 20, 30, 40])
        
    def test_reshape_to_spatial_grid_basic(self):
        """Test basic reshape operation"""
        reshaped = reshape_to_spatial_grid(self.flattened_data, self.id_matrix)
    
        # After flip_y=True, check the bottom row instead of top row
        # Check shape
        self.assertEqual(reshaped.shape, self.super_mask.shape)
    
        # Check values are placed correctly (accounting for flip)
        # The values are flipped, so check the bottom row
        self.assertFalse(np.isnan(reshaped[1, 0]))  # Bottom left (was top left)
        self.assertFalse(np.isnan(reshaped[1, 1]))  # Bottom middle (was top middle)
        self.assertTrue(np.isnan(reshaped[1, 2]))   # Bottom right should be NaN
        self.assertFalse(np.isnan(reshaped[0, 1]))  # Top middle
        self.assertFalse(np.isnan(reshaped[0, 2]))  # Top right
    
    def test_reshape_to_spatial_grid_no_flip(self):
        """Test reshape without flipping"""
        reshaped = reshape_to_spatial_grid(self.flattened_data, self.id_matrix, flip_y=False)
        
        # Should have same shape
        self.assertEqual(reshaped.shape, self.super_mask.shape)
        
        # Values should be in correct positions
        expected_valid_cells = np.sum(self.super_mask)
        actual_valid_cells = np.sum(~np.isnan(reshaped))
        self.assertEqual(actual_valid_cells, expected_valid_cells)
    
    def test_reshape_all_to_spatial_grid(self):
        """Test reshaping multiple flattened arrays"""
        flattened_dict = {
            'data1': self.flattened_data,
            'data2': self.flattened_data * 2
        }
        
        reshaped_dict = reshape_all_to_spatial_grid(flattened_dict, self.id_matrix)
        
        # Check all matrices were reshaped
        for key, matrix in reshaped_dict.items():
            self.assertEqual(matrix.shape, self.super_mask.shape)
        
        # Check values
        self.assertTrue(np.all(reshaped_dict['data2'][~np.isnan(reshaped_dict['data2'])] == 
                              2 * reshaped_dict['data1'][~np.isnan(reshaped_dict['data1'])]))


class TestStacker(unittest.TestCase):
    """Test stacking functions"""
    
    def setUp(self):
        """Create test data for stacking tests"""
        self.flattened_dict = {
            'time1': np.array([[1, 2, 3, 4]]),
            'time2': np.array([[5, 6, 7, 8]]),
            'time3': np.array([[9, 10, 11, 12]])
        }
    
    def test_create_super_matrix_default_order(self):
        """Test super matrix creation with default ordering"""
        super_matrix, keys = create_super_matrix(self.flattened_dict)
        
        # Check shape
        self.assertEqual(super_matrix.shape, (3, 4))
        
        # Check keys are sorted
        self.assertEqual(keys, ['time1', 'time2', 'time3'])
        
        # Check values
        np.testing.assert_array_equal(super_matrix[0], [1, 2, 3, 4])
        np.testing.assert_array_equal(super_matrix[1], [5, 6, 7, 8])
        np.testing.assert_array_equal(super_matrix[2], [9, 10, 11, 12])
    
    def test_create_super_matrix_custom_order(self):
        """Test super matrix creation with custom ordering"""
        custom_keys = ['time3', 'time1', 'time2']
        super_matrix, keys = create_super_matrix(self.flattened_dict, keys=custom_keys)
        
        # Check returned keys match input
        self.assertEqual(keys, custom_keys)
        
        # Check values are in correct order
        np.testing.assert_array_equal(super_matrix[0], [9, 10, 11, 12])
        np.testing.assert_array_equal(super_matrix[1], [1, 2, 3, 4])
        np.testing.assert_array_equal(super_matrix[2], [5, 6, 7, 8])
    
    def test_create_super_matrix_dimension_mismatch(self):
        """Test error handling for dimension mismatch"""
        bad_dict = {
            'time1': np.array([[1, 2, 3]]),
            'time2': np.array([[4, 5, 6, 7]])  # Different length
        }
        
        with self.assertRaises(DimensionError):
            create_super_matrix(bad_dict)


class TestIO(unittest.TestCase):
    """Test I/O functions"""
    
    def setUp(self):
        """Create test data for I/O tests"""
        self.test_dir = tempfile.mkdtemp()
        self.test_files = []
        
        # Create test NetCDF files (place the code HERE, inside setUp)
        for i in range(3):
            filename = f"test_202{i+1}01.nc"
            filepath = os.path.join(self.test_dir, filename)
            
            with nc.Dataset(filepath, 'w') as ds:
                # Create dimensions
                ds.createDimension('x', 10)
                ds.createDimension('y', 8)
                ds.createDimension('two', 2)  # Create dimension for size 2
                
                # Create variables
                z = ds.createVariable('z', 'f4', ('y', 'x'))
                x_range = ds.createVariable('x_range', 'f4', ('two',))
                y_range = ds.createVariable('y_range', 'f4', ('two',))
                dimension = ds.createVariable('dimension', 'f4', ('two',))
                spacing = ds.createVariable('spacing', 'f4', ('two',))
                
                # Fill with test data
                z[:] = np.random.rand(8, 10)
                x_range[:] = [0, 10]
                y_range[:] = [0, 8]
                dimension[:] = [10, 8]
                spacing[:] = [1, 1]
            
            self.test_files.append(filepath)
    
    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.test_dir)
    
    
    def test_read_netcdf_valid_file(self):
        """Test reading a valid NetCDF file"""
        data = read_netcdf(self.test_files[0])
        
        # Check required keys
        self.assertIn('z', data)
        self.assertIn('longitude', data)
        self.assertIn('latitude', data)
        self.assertIn('dimensions', data)
        self.assertIn('spacing', data)
        
        # Check data shapes
        self.assertEqual(data['z'].shape, (8, 10))
        self.assertEqual(data['longitude'].shape, (8, 10))
        self.assertEqual(data['latitude'].shape, (8, 10))
    
    def test_read_netcdf_invalid_file(self):
        """Test error handling for invalid files"""
        with self.assertRaises(FileReadError):
            read_netcdf("nonexistent.nc")
    
    def test_sort_files_by_date(self):
        """Test sorting files by date"""
        sorted_files = sort_files_by_date(self.test_dir, '.nc', r'(\d{6})', '%Y%m')
        
        # Check order is correct
        self.assertEqual(len(sorted_files), 3)
        self.assertTrue('202101' in sorted_files[0])
        self.assertTrue('202201' in sorted_files[1])
        self.assertTrue('202301' in sorted_files[2])


class TestUtils(unittest.TestCase):
    """Test utility functions"""
    
    def test_extract_date_from_filename_basic(self):
        """Test basic date extraction"""
        filename = "data_202301.nc"
        pattern = r"(\d{6})"
        format_string = "%Y%m"
        
        date = extract_date_from_filename(filename, pattern, format_string)
        
        self.assertIsInstance(date, datetime)
        self.assertEqual(date.year, 2023)
        self.assertEqual(date.month, 1)
    
    def test_extract_date_from_filename_no_pattern(self):
        """Test date extraction without pattern"""
        filename = "202301.nc"
        
        date = extract_date_from_filename(filename)
        
        # Should return int
        self.assertEqual(date, 202301)
    
    def test_filter_files_by_date_range(self):
        """Test filtering files by date range"""
        files = [
            "data_202301.nc",
            "data_202302.nc",
            "data_202303.nc",
            "data_202304.nc"
        ]
        
        filtered = filter_files_by_date_range(
            files,
            start_date="2023-02-01",
            end_date="2023-03-31",
            date_pattern=r"(\d{6})",
            date_format="%Y%m"
        )
        
        # Should include February and March
        self.assertEqual(len(filtered), 2)
        self.assertTrue(any("202302" in f for f in filtered))
        self.assertTrue(any("202303" in f for f in filtered))


# Run all tests
if __name__ == '__main__':
    unittest.main()