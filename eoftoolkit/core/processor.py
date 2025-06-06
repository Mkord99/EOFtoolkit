"""Main processor class for EOFtoolkit."""

import os
import numpy as np
import pandas as pd
from datetime import datetime

from eoftoolkit.core.exceptions import EOFToolkitError, FileReadError, DimensionError
from eoftoolkit.io.sorter import sort_files_by_date
from eoftoolkit.io.reader import read_netcdf
from eoftoolkit.processor.dimensions import standardize_dimensions
from eoftoolkit.processor.masking import create_binary_mask, create_super_mask
from eoftoolkit.processor.identification import create_id_matrix, get_id_coordinates
from eoftoolkit.processor.flattener import flatten_matrices, center_matrices
from eoftoolkit.processor.stacker import create_super_matrix
from eoftoolkit.analysis.svd import perform_svd
from eoftoolkit.analysis.reconstruction import reconstruct_from_modes


class EOFProcessor:
    """
    Main processor class for EOF analysis of NetCDF files.
    
    This class handles the entire EOF analysis pipeline:
    1. Reading NetCDF files
    2. Standardizing dimensions
    3. Creating binary masks and super mask
    4. Creating ID matrix
    5. Flattening matrices
    6. Creating super matrix
    7. Performing SVD analysis
    8. Reconstructing data
    
    Parameters
    ----------
    verbose : bool, optional
        Whether to print progress messages. Default is True.
    """
    
    def __init__(self, verbose=True, projection='merc', projection_params=None):
        """
        Initialize EOFProcessor.
        
        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        projection : str, optional
            Map projection to use for visualization. Default is 'merc' (Mercator).
        projection_params : dict, optional
            Dictionary of projection parameters for visualization.
        """
        self.verbose = verbose
        self.projection = projection
        self.projection_params = projection_params or {}
        self.reset()
    
    def reset(self):
        """Reset all instance variables."""
        self.file_paths = None
        self.file_keys = None
        self.data_dict = None
        self.standardized_data = None
        self.target_dims = None
        self.mask_dict = None
        self.super_mask = None
        self.id_matrix = None
        self.id_coordinates = None
        self.longitude = None
        self.latitude = None
        self.flattened_data = None
        self.flattened_id_matrix = None
        self.centered_data = None
        self.mean_dict = None
        self.super_matrix = None
        self.svd_results = None
        self.reconstruction_results = None
        # Keep projection settings
        # self.projection and self.projection_params are preserved
    
    def process_directory(self, directory_path, file_extension='.nc', 
                        date_pattern=None, date_format=None,
                        start_date=None, end_date=None):
        """
        Process a directory of NetCDF files.
        
        Parameters
        ----------
        directory_path : str
            Path to the directory containing NetCDF files.
        file_extension : str, optional
            Extension of files to process. Default is '.nc'.
        date_pattern : str, optional
            Regular expression pattern to extract date from filename.
        date_format : str, optional
            Format string for parsing the date if a pattern is provided.
        start_date : str or datetime, optional
            Start date for filtering files.
        end_date : str or datetime, optional
            End date for filtering files.
            
        Returns
        -------
        dict
            Processing results containing super_matrix, id_matrix, etc.
        """
        # Sort files by date
        if self.verbose:
            print("Sorting files by date...")
        
        self.file_paths = sort_files_by_date(
            directory_path, file_extension, date_pattern, date_format
        )
        
        if not self.file_paths:
            raise FileReadError(f"No {file_extension} files found in {directory_path}")
        
        # Filter files by date range if provided
        if start_date is not None and end_date is not None:
            from eoftoolkit.core.utils import filter_files_by_date_range
            
            self.file_paths = filter_files_by_date_range(
                self.file_paths, start_date, end_date, date_pattern, date_format
            )
            
            if not self.file_paths:
                raise FileReadError(f"No files found in date range {start_date} to {end_date}")
        
        # Extract file keys (basenames without extension)
        self.file_keys = [os.path.splitext(os.path.basename(fp))[0] for fp in self.file_paths]
        
        # Read NetCDF files
        if self.verbose:
            print(f"Reading {len(self.file_paths)} NetCDF files...")
        
        self.data_dict = {}
        
        for i, (file_path, file_key) in enumerate(zip(self.file_paths, self.file_keys)):
            if self.verbose:
                from eoftoolkit.core.utils import print_progress
                print_progress(i+1, len(self.file_paths), prefix='Reading files:', suffix='Complete')
            
            # Read file
            try:
                data = read_netcdf(file_path)
                self.data_dict[file_key] = data
                
                # Store longitude and latitude grids from the first file
                if i == 0:
                    self.longitude = data['longitude']
                    self.latitude = data['latitude']
            
            except Exception as e:
                raise FileReadError(f"Error reading file {file_path}: {str(e)}")
        
        # Standardize dimensions
        if self.verbose:
            print("Standardizing dimensions...")
        
        # Extract 'z' values from data_dict
        z_dict = {key: data['z'] for key, data in self.data_dict.items()}
        
        # Standardize dimensions
        self.standardized_data, self.target_dims = standardize_dimensions(z_dict)
        
        # Create binary masks
        if self.verbose:
            print("Creating binary masks...")
        
        self.mask_dict = {}
        for key, matrix in self.standardized_data.items():
            self.mask_dict[key] = create_binary_mask(matrix)
        
        # Create super mask
        if self.verbose:
            print("Creating super mask...")
        
        self.super_mask = create_super_mask(self.mask_dict)
        
        # Create ID matrix
        if self.verbose:
            print("Creating ID matrix...")
        
        self.id_matrix = create_id_matrix(self.super_mask)
        self.id_coordinates = get_id_coordinates(self.id_matrix, self.longitude, self.latitude)
        
        # Flatten matrices
        if self.verbose:
            print("Flattening matrices...")
        
        self.flattened_data, self.flattened_id_matrix = flatten_matrices(
            self.standardized_data, self.id_matrix, self.super_mask
        )
        
        # Center matrices
        if self.verbose:
            print("Centering matrices...")
        
        self.centered_data, self.mean_dict = center_matrices(
            self.flattened_data, axis=1, return_means=True
        )
        
        # Create super matrix
        if self.verbose:
            print("Creating super matrix...")
        
        self.super_matrix, _ = create_super_matrix(self.centered_data, keys=self.file_keys)
        
        if self.verbose:
            print("Processing complete.")
        
        # Return processing results
        return {
            'super_matrix': self.super_matrix,
            'id_matrix': self.id_matrix,
            'super_mask': self.super_mask,
            'longitude': self.longitude,
            'latitude': self.latitude,
            'file_keys': self.file_keys,
            'target_dims': self.target_dims
        }
    
    def perform_svd(self, num_modes=None, compute_surfaces=True):
        """
        Perform SVD analysis on the super matrix.
        
        Parameters
        ----------
        num_modes : int, optional
            Number of modes to extract. If None, extracts all modes.
        compute_surfaces : bool, optional
            Whether to compute corresponding surfaces. Default is True.
            
        Returns
        -------
        dict
            SVD results containing EOFs, PCs, etc.
        """
        if self.super_matrix is None:
            raise EOFToolkitError(
                "Super matrix is not available. Run process_directory first."
            )
        
        if self.verbose:
            print(f"Performing SVD analysis{'' if num_modes is None else f' with {num_modes} modes'}...")
        
        # Perform SVD
        self.svd_results = perform_svd(self.super_matrix, num_modes, compute_surfaces)
        
        # Add super_matrix to results for reconstruction
        self.svd_results['super_matrix'] = self.super_matrix
        
        if self.verbose:
            print("SVD analysis complete.")
            
            # Print variance explained
            variance = self.svd_results['explained_variance']
            cumulative = self.svd_results['cumulative_variance']
            
            print("\nVariance Explained:")
            for i, (var, cum) in enumerate(zip(variance, cumulative)):
                print(f"  Mode {i+1}: {var:.2f}% (Cumulative: {cum:.2f}%)")
        
        return self.svd_results
    
    def reconstruct(self, max_modes=None, metric='rmse'):
        """
        Reconstruct data from SVD results.
        
        Parameters
        ----------
        max_modes : int, optional
            Maximum number of modes to use in reconstruction.
            If None, uses all available modes.
        metric : str, optional
            Metric to use for determining optimal reconstruction.
            Default is 'rmse'.
            
        Returns
        -------
        dict
            Reconstruction results.
        """
        if self.svd_results is None:
            raise EOFToolkitError(
                "SVD results are not available. Run perform_svd first."
            )
        
        if self.verbose:
            print(f"Performing data reconstruction with up to {max_modes or 'all'} modes...")
        
        # Reconstruct data
        self.reconstruction_results = reconstruct_from_modes(
            self.svd_results, max_modes, metric
        )
        
        if self.verbose:
            print("Reconstruction complete.")
            
            # Print optimal reconstruction
            optimal_modes = self.reconstruction_results['optimal_mode_count']
            error = self.reconstruction_results['error_metrics'][optimal_modes][metric]
            
            print(f"\nOptimal reconstruction uses {optimal_modes} modes")
            print(f"{metric.upper()}: {error:.4f}")
        
        return self.reconstruction_results
    
    def get_eof(self, mode_number, reshape=True):
        """
        Get a specific EOF.
        
        Parameters
        ----------
        mode_number : int
            Mode number (1-based).
        reshape : bool, optional
            Whether to reshape the EOF to a 2D grid. Default is True.
            
        Returns
        -------
        ndarray
            EOF values.
        """
        if self.svd_results is None:
            raise EOFToolkitError(
                "SVD results are not available. Run perform_svd first."
            )
        
        # Adjust for 0-based indexing
        idx = mode_number - 1
        
        if idx < 0 or idx >= self.svd_results['eofs'].shape[0]:
            raise IndexError(f"Mode number {mode_number} is out of range")
        
        # Get EOF
        eof = self.svd_results['eofs'][idx, :]
        
        # Reshape if requested
        if reshape:
            from eoftoolkit.processor.reshaper import reshape_to_spatial_grid
            eof = reshape_to_spatial_grid(eof, self.id_matrix, self.target_dims)
        
        return eof
    
    def get_pc(self, mode_number):
        """
        Get a specific PC.
        
        Parameters
        ----------
        mode_number : int
            Mode number (1-based).
            
        Returns
        -------
        ndarray
            PC values.
        """
        if self.svd_results is None:
            raise EOFToolkitError(
                "SVD results are not available. Run perform_svd first."
            )
        
        # Adjust for 0-based indexing
        idx = mode_number - 1
        
        if idx < 0 or idx >= self.svd_results['pcs'].shape[1]:
            raise IndexError(f"Mode number {mode_number} is out of range")
        
        # Get PC
        pc = self.svd_results['pcs'][:, idx]
        
        return pc
    
    def get_reconstruction(self, mode_count=None, timestamp_index=0, reshape=True):
        """
        Get a specific reconstruction.
        
        Parameters
        ----------
        mode_count : int, optional
            Number of modes to use. If None, uses the optimal number.
        timestamp_index : int, optional
            Index of the timestamp to get. Default is 0.
        reshape : bool, optional
            Whether to reshape the reconstruction to a 2D grid. Default is True.
            
        Returns
        -------
        ndarray
            Reconstruction values.
        """
        if self.reconstruction_results is None:
            raise EOFToolkitError(
                "Reconstruction results are not available. Run reconstruct first."
            )
        
        # Determine which reconstruction to use
        if mode_count is None:
            mode_count = self.reconstruction_results['optimal_mode_count']
        
        if mode_count not in self.reconstruction_results['reconstructions']:
            raise IndexError(f"No reconstruction available with {mode_count} modes")
        
        # Get reconstruction
        reconstruction = self.reconstruction_results['reconstructions'][mode_count]
        
        # Extract specific timestamp if multiple
        if len(reconstruction.shape) > 1:
            reconstruction = reconstruction[timestamp_index, :]
        
        # Reshape if requested
        if reshape:
            from eoftoolkit.processor.reshaper import reshape_to_spatial_grid
            reconstruction = reshape_to_spatial_grid(reconstruction, self.id_matrix, self.target_dims)
        
        return reconstruction
    
    def get_original_data(self, timestamp_index=0, reshape=True):
        """
        Get original data for a specific timestamp.
        
        Parameters
        ----------
        timestamp_index : int, optional
            Index of the timestamp to get. Default is 0.
        reshape : bool, optional
            Whether to reshape the data to a 2D grid. Default is True.
            
        Returns
        -------
        ndarray
            Original data values.
        """
        if self.super_matrix is None:
            raise EOFToolkitError(
                "Super matrix is not available. Run process_directory first."
            )
        
        # Get original data
        original = self.super_matrix[timestamp_index, :]
        
        # Reshape if requested
        if reshape:
            from eoftoolkit.processor.reshaper import reshape_to_spatial_grid
            original = reshape_to_spatial_grid(original, self.id_matrix, self.target_dims)
        
        return original
    
    def get_dates(self, as_datetime=False, date_format=None):
        """
        Get dates for the processed files.
        
        Parameters
        ----------
        as_datetime : bool, optional
            Whether to convert dates to datetime objects. Default is False.
        date_format : str, optional
            Format string for parsing dates. Default is None.
            
        Returns
        -------
        list
            List of dates.
        """
        if self.file_keys is None:
            raise EOFToolkitError(
                "File keys are not available. Run process_directory first."
            )
        
        if as_datetime and date_format:
            dates = []
            for key in self.file_keys:
                try:
                    date = datetime.strptime(key, date_format)
                    dates.append(date)
                except ValueError:
                    dates.append(key)
            return dates
        else:
            return self.file_keys
    
    def visualize_eof(self, mode_number, projection=None, **kwargs):
        """
        Visualize a specific EOF.
    
        Parameters
        ----------
        mode_number : int
            Mode number (1-based).
        projection : str, optional
            Map projection to use. If None, uses the configured projection.
        **kwargs : dict
            Additional parameters to pass to plot_eof.
        
        Returns
        -------
        tuple
            (fig, ax) tuple with the figure and axes objects.
        """
        from eoftoolkit.visualization.eof_plots import plot_eof
    
        # Get EOF
        eof = self.get_eof(mode_number, reshape=False)
    
        # Use configured projection if not specified
        proj_params = self.get_projection_params()
        if projection is None:
            projection = proj_params['projection']
            # Merge projection params
            for k, v in proj_params['params'].items():
                if k not in kwargs:
                    kwargs[k] = v
    
        # Plot EOF
        return plot_eof(
            eof=eof,
            id_matrix=self.id_matrix,
            lats=self.latitude,
            lons=self.longitude,
            mode_number=mode_number,
            projection=projection,
            **kwargs
        )   
    
    def visualize_pc(self, mode_number, dates=None, date_format=None, **kwargs):
        """
        Visualize a specific PC.
        
        Parameters
        ----------
        mode_number : int
            Mode number (1-based).
        dates : list, optional
            Dates or time values. If None, uses file keys.
        date_format : str, optional
            Format string for parsing dates (e.g., '%Y%m' for '202301').
            If None, auto-detection is attempted.
        **kwargs : dict
            Additional parameters to pass to plot_pc.
            
        Returns
        -------
        tuple
            (fig, ax) tuple with the figure and axes objects.
        """
        from eoftoolkit.visualization.pc_plots import plot_pc
        
        # Get PC
        pc = self.get_pc(mode_number)
        
        # Use file keys as dates if not provided
        if dates is None:
            dates = self.file_keys
        
        # Add date_format to kwargs if provided
        if date_format is not None and 'date_format' not in kwargs:
            kwargs['date_format'] = date_format
        
        # Plot PC
        return plot_pc(
            pc=pc,
            dates=dates,
            mode_number=mode_number,
            **kwargs
        )
    
    def visualize_reconstruction(self, mode_count=None, timestamp_index=0, projection=None, **kwargs):
        """
        Visualize a specific reconstruction.
        
        Parameters
        ----------
        mode_count : int, optional
            Number of modes to use. If None, uses the optimal number.
        timestamp_index : int, optional
            Index of the timestamp to visualize. Default is 0.
        projection : str, optional
            Map projection to use. If None, uses the configured projection.
        **kwargs : dict
            Additional parameters to pass to plot_reconstruction.
            
        Returns
        -------
        tuple
            (fig, ax) tuple with the figure and axes objects.
        """
        from eoftoolkit.visualization.reconstruction_plots import plot_reconstruction
        
        # Get reconstruction
        reconstruction = self.get_reconstruction(mode_count, timestamp_index, reshape=False)
        
        # Use configured projection if not specified
        proj_params = self.get_projection_params()
        if projection is None:
            projection = proj_params['projection']
            # Merge projection params
            for k, v in proj_params['params'].items():
                if k not in kwargs:
                    kwargs[k] = v
        
        # Plot reconstruction
        return plot_reconstruction(
            reconstruction=reconstruction,
            id_matrix=self.id_matrix,
            lats=self.latitude,
            lons=self.longitude,
            timestamp_index=timestamp_index,
            projection=projection,
            **kwargs
        )
    
    def visualize_reconstruction_error(self, metric='rmse', **kwargs):
        """
        Visualize reconstruction error metrics.
        
        Parameters
        ----------
        metric : str, optional
            Metric to visualize. Default is 'rmse'.
        **kwargs : dict
            Additional parameters to pass to plot_reconstruction_error.
            
        Returns
        -------
        tuple
            (fig, ax) tuple with the figure and axes objects.
        """
        from eoftoolkit.visualization.error_plots import plot_reconstruction_error
        
        if self.reconstruction_results is None:
            raise EOFToolkitError(
                "Reconstruction results are not available. Run reconstruct first."
            )
        
        # Plot error metrics
        return plot_reconstruction_error(
            error_metrics=self.reconstruction_results['error_metrics'],
            metric_name=metric,
            **kwargs
        )
    
    def visualize_comparison(self, mode_count=None, timestamp_index=0, projection=None, **kwargs):
        """
        Visualize comparison between original and reconstructed data.
        
        Parameters
        ----------
        mode_count : int, optional
            Number of modes to use. If None, uses the optimal number.
        timestamp_index : int, optional
            Index of the timestamp to visualize. Default is 0.
        projection : str, optional
            Map projection to use. If None, uses the configured projection.
        **kwargs : dict
            Additional parameters to pass to plot_reconstruction_comparison.
            
        Returns
        -------
        tuple
            (fig, axes) tuple with the figure and axes objects.
        """
        from eoftoolkit.visualization.spatial import plot_reconstruction_comparison
        
        # Get original and reconstructed data
        original = self.get_original_data(timestamp_index, reshape=False)
        reconstruction = self.get_reconstruction(mode_count, timestamp_index, reshape=False)
        
        # Use configured projection if not specified
        proj_params = self.get_projection_params()
        if projection is None:
            projection = proj_params['projection']
            # Merge projection params
            for k, v in proj_params['params'].items():
                if k not in kwargs:
                    kwargs[k] = v
        
        # Plot comparison
        return plot_reconstruction_comparison(
            original=original,
            reconstructed=reconstruction,
            id_matrix=self.id_matrix,
            lats=self.latitude,
            lons=self.longitude,
            timestamp_index=timestamp_index,
            projection=projection,
            **kwargs
        )
    
    def save_results(self, output_dir, prefix='eof_analysis'):
        """
        Save analysis results to files.
        
        Parameters
        ----------
        output_dir : str
            Directory to save the files.
        prefix : str, optional
            Prefix for the output filenames.
            
        Returns
        -------
        dict
            Dictionary with paths to saved files.
        """
        from eoftoolkit.io.writer import save_results
        
        # Prepare results dictionary
        results = {
            'super_matrix': self.super_matrix,
            'id_matrix': self.id_matrix,
            'super_mask': self.super_mask
        }
        
        # Add SVD results if available
        if self.svd_results is not None:
            results.update({
                'eofs': self.svd_results['eofs'],
                'pcs': self.svd_results['pcs'],
                'singular_values': self.svd_results['singular_values'],
                'explained_variance': self.svd_results['explained_variance']
            })
        
        # Add reconstruction results if available
        if self.reconstruction_results is not None:
            results.update({
                'reconstruction': self.reconstruction_results['optimal_reconstruction'],
                'error_metrics': self.reconstruction_results['error_metrics']
            })
        
        # Save results
        return save_results(results, output_dir, prefix)

        
    def configure_projection(self, projection='merc', projection_params=None):
        """
        Configure projection system for visualization.
        
        Parameters
        ----------
        projection : str, optional
            Map projection to use. Default is 'merc' (Mercator).
            Options include 'merc', 'lcc' (Lambert Conformal Conic), 'stere', etc.
        projection_params : dict, optional
            Dictionary of projection parameters.
            For 'lcc', this could include 'lat_1', 'lat_2', 'lat_0', 'lon_0', etc.
            
        Returns
        -------
        dict
            Current projection configuration.
        """
        # Set default projection parameters if not provided
        if projection_params is None:
            projection_params = {}
        
        # Set automatic default parameters based on data extent
        if self.latitude is not None and self.longitude is not None:
            lat_min = np.nanmin(self.latitude)
            lat_max = np.nanmax(self.latitude)
            lon_min = np.nanmin(self.longitude)
            lon_max = np.nanmax(self.longitude)
            
            # Set defaults for different projections
            if projection == 'lcc' and 'lat_0' not in projection_params:
                # Lambert Conformal Conic defaults
                projection_params.setdefault('lat_1', lat_min)
                projection_params.setdefault('lat_2', lat_max)
                projection_params.setdefault('lat_0', (lat_min + lat_max) / 2)
                projection_params.setdefault('lon_0', (lon_min + lon_max) / 2)
            
            elif projection == 'stere' and 'lat_0' not in projection_params:
                # Stereographic projection defaults
                projection_params.setdefault('lat_0', (lat_min + lat_max) / 2)
                projection_params.setdefault('lon_0', (lon_min + lon_max) / 2)
        
        # Store projection configuration
        self.projection = projection
        self.projection_params = projection_params
        
        if self.verbose:
            print(f"Configured projection: {projection}")
            if projection_params:
                params_str = ", ".join(f"{k}={v}" for k, v in projection_params.items())
                print(f"Projection parameters: {params_str}")
        
        return {'projection': projection, 'params': projection_params}
    
    def get_projection_params(self):
        """
        Get current projection configuration.
        
        Returns
        -------
        dict
            Current projection configuration.
        """
        # Configure default projection if not already configured
        if not hasattr(self, 'projection') or not hasattr(self, 'projection_params'):
            self.configure_projection()
        
        return {'projection': self.projection, 'params': self.projection_params}
