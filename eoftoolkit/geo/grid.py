"""Module for handling geographic grids."""

import numpy as np


def create_grid(lon_range, lat_range, lon_step, lat_step):
    """
    Create a regular geographic grid.
    
    Parameters
    ----------
    lon_range : tuple
        Longitude range as (lon_min, lon_max).
    lat_range : tuple
        Latitude range as (lat_min, lat_max).
    lon_step : float
        Longitude step size.
    lat_step : float
        Latitude step size.
        
    Returns
    -------
    tuple
        (lons, lats) tuple with 2D grids of longitude and latitude values.
    """
    # Create 1D arrays
    lons = np.arange(lon_range[0], lon_range[1] + lon_step/2, lon_step)
    lats = np.arange(lat_range[0], lat_range[1] + lat_step/2, lat_step)
    
    # Create 2D grids
    lons_grid, lats_grid = np.meshgrid(lons, lats)
    
    return lons_grid, lats_grid


def get_grid_info(lons, lats):
    """
    Get information about a geographic grid.
    
    Parameters
    ----------
    lons : ndarray
        2D grid of longitude values.
    lats : ndarray
        2D grid of latitude values.
        
    Returns
    -------
    dict
        Dictionary with grid information.
    """
    lon_range = (np.nanmin(lons), np.nanmax(lons))
    lat_range = (np.nanmin(lats), np.nanmax(lats))
    
    # Try to determine step sizes
    if lons.shape[1] > 1:
        lon_steps = np.diff(lons[0, :])
        lon_step = np.median(lon_steps)
    else:
        lon_step = None
    
    if lats.shape[0] > 1:
        lat_steps = np.diff(lats[:, 0])
        lat_step = np.median(lat_steps)
    else:
        lat_step = None
    
    return {
        'lon_range': lon_range,
        'lat_range': lat_range,
        'lon_step': lon_step,
        'lat_step': lat_step,
        'shape': lons.shape
    }


def interpolate_to_grid(data, lons, lats, target_lons, target_lats, method='linear'):
    """
    Interpolate data to a different grid.
    
    Parameters
    ----------
    data : ndarray
        2D data array.
    lons : ndarray
        2D grid of longitude values for data.
    lats : ndarray
        2D grid of latitude values for data.
    target_lons : ndarray
        2D grid of longitude values for target grid.
    target_lats : ndarray
        2D grid of latitude values for target grid.
    method : str, optional
        Interpolation method. Default is 'linear'.
        
    Returns
    -------
    ndarray
        Interpolated data on target grid.
    """
    from scipy.interpolate import griddata
    
    # Flatten input grids
    lons_flat = lons.flatten()
    lats_flat = lats.flatten()
    data_flat = data.flatten()
    
    # Remove NaN values
    valid = ~np.isnan(data_flat)
    lons_valid = lons_flat[valid]
    lats_valid = lats_flat[valid]
    data_valid = data_flat[valid]
    
    # Stack coordinates
    coords = np.column_stack((lons_valid, lats_valid))
    
    # Flatten target grids
    target_lons_flat = target_lons.flatten()
    target_lats_flat = target_lats.flatten()
    target_coords = np.column_stack((target_lons_flat, target_lats_flat))
    
    # Interpolate
    target_data_flat = griddata(coords, data_valid, target_coords, method=method, fill_value=np.nan)
    
    # Reshape to target grid
    target_data = target_data_flat.reshape(target_lons.shape)
    
    return target_data