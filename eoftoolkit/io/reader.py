"""Module for reading NetCDF files."""

import netCDF4 as nc
import numpy as np
from eoftoolkit.core.exceptions import FileReadError


def read_netcdf(file_path):
    """
    Read a NetCDF file and extract its content.
    
    Parameters
    ----------
    file_path : str
        Path to the NetCDF file.
        
    Returns
    -------
    dict
        Dictionary containing extracted data with the following keys:
        - 'z': The data values (masked array)
        - 'longitude': 2D grid of longitude values
        - 'latitude': 2D grid of latitude values
        - 'dimensions': Original dimensions of the data
        - 'spacing': Grid spacing information
    """
    try:
        data = nc.Dataset(file_path, 'r')
        
        # Extract the main variable (assuming 'z' by default)
        z_var_name = 'z'
        if z_var_name not in data.variables:
            # Try to find the main variable if 'z' is not available
            exclude_vars = ['x', 'y', 'lat', 'lon', 'latitude', 'longitude', 
                           'x_range', 'y_range', 'dimension', 'spacing', 'time']
            data_vars = [v for v in data.variables if v not in exclude_vars]
            if data_vars:
                z_var_name = data_vars[0]
            else:
                raise FileReadError(f"Could not find main data variable in {file_path}")
        
        z_data = data.variables[z_var_name][:]
        
        # Extract coordinate information
        lon_range = data.variables['x_range'][:] if 'x_range' in data.variables else None
        lat_range = data.variables['y_range'][:] if 'y_range' in data.variables else None
        dim = data.variables['dimension'][:] if 'dimension' in data.variables else None
        space = data.variables['spacing'][:] if 'spacing' in data.variables else None
        
        # Create coordinate grids
        if lon_range is not None and lat_range is not None and dim is not None and space is not None:
            lons = np.linspace(lon_range[0] + space[0]/2, lon_range[1] - space[0]/2, num=int(dim[0]))
            lats = np.linspace(lat_range[1] - space[1]/2, lat_range[0] + space[1]/2, num=int(dim[1]))
            lons_grid, lats_grid = np.meshgrid(lons, lats)
            
            # Reshape values to dimensions of the grid
            # This is the critical part - reshape according to dim values
            if len(z_data.shape) == 1:  # If z_data is 1D
                z_data = np.reshape(z_data, (int(dim[1]), int(dim[0])))
        else:
            # If coordinate information is not available
            if len(z_data.shape) == 1:
                # Can't properly reshape without dimensions, but we can make a row vector
                z_data = z_data.reshape(1, -1)
                lons_grid, lats_grid = np.meshgrid(np.arange(z_data.shape[1]), np.arange(1))
            else:
                lons_grid, lats_grid = np.meshgrid(np.arange(z_data.shape[1]), np.arange(z_data.shape[0]))
        
        # Mask invalid values
        z = np.ma.masked_invalid(z_data)
        
        return {
            "z": z,
            "longitude": lons_grid,
            "latitude": lats_grid,
            "dimensions": dim,
            "spacing": space
        }
    
    except Exception as e:
        raise FileReadError(f"Error reading NetCDF file {file_path}: {str(e)}")