"""Module for handling map projections and coordinate transformations."""

try:
    from pyproj import Proj, Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

import numpy as np
from eoftoolkit.core.exceptions import EOFToolkitError


def check_pyproj():
    """Check if pyproj is available."""
    if not PYPROJ_AVAILABLE:
        raise EOFToolkitError(
            "The pyproj library is required for projection operations. "
            "Please install it using pip: pip install pyproj"
        )


def create_projection(proj_type='lcc', lat_range=None, lon_range=None, **kwargs):
    """
    Create a map projection.
    
    Parameters
    ----------
    proj_type : str, optional
        Projection type. Default is 'lcc' (Lambert Conformal Conic).
    lat_range : tuple, optional
        Latitude range as (lat_min, lat_max). Required for some projections.
    lon_range : tuple, optional
        Longitude range as (lon_min, lon_max). Required for some projections.
    **kwargs : dict
        Additional projection parameters.
        
    Returns
    -------
    Proj
        Projection object.
    """
    check_pyproj()
    
    # Set up projection parameters
    if proj_type == 'lcc':
        # Lambert Conformal Conic projection
        if lat_range is None or lon_range is None:
            raise ValueError("lat_range and lon_range are required for LCC projection")
            
        lat1 = lat_range[0]  # Lower standard parallel
        lat2 = lat_range[1]  # Upper standard parallel
        lon0 = (lon_range[0] + lon_range[1]) / 2  # Central longitude
        lat0 = (lat_range[0] + lat_range[1]) / 2  # Central latitude
        
        proj_params = {
            'proj': 'lcc',
            'lat_1': lat1,
            'lat_2': lat2,
            'lat_0': lat0,
            'lon_0': lon0,
            'x_0': 0,
            'y_0': 0,
            'ellps': 'WGS84'
        }
        
    elif proj_type == 'merc':
        # Mercator projection
        proj_params = {
            'proj': 'merc',
            'ellps': 'WGS84'
        }
        
    elif proj_type == 'tmerc':
        # Transverse Mercator projection
        if lon_range is None:
            raise ValueError("lon_range is required for Transverse Mercator projection")
            
        lon0 = (lon_range[0] + lon_range[1]) / 2  # Central longitude
        
        proj_params = {
            'proj': 'tmerc',
            'lon_0': lon0,
            'x_0': 0,
            'y_0': 0,
            'ellps': 'WGS84'
        }
        
    elif proj_type == 'stere':
        # Stereographic projection
        if lat_range is None or lon_range is None:
            raise ValueError("lat_range and lon_range are required for Stereographic projection")
            
        lat0 = (lat_range[0] + lat_range[1]) / 2  # Central latitude
        lon0 = (lon_range[0] + lon_range[1]) / 2  # Central longitude
        
        proj_params = {
            'proj': 'stere',
            'lat_0': lat0,
            'lon_0': lon0,
            'x_0': 0,
            'y_0': 0,
            'ellps': 'WGS84'
        }
        
    else:
        # Generic projection
        proj_params = {'proj': proj_type, 'ellps': 'WGS84'}
    
    # Update with any additional parameters
    proj_params.update(kwargs)
    
    # Create projection
    proj = Proj(**proj_params)
    
    return proj


def transform_coordinates(lons, lats, src_proj='epsg:4326', dst_proj=None):
    """
    Transform coordinates from one projection to another.
    
    Parameters
    ----------
    lons : ndarray
        Longitude values.
    lats : ndarray
        Latitude values.
    src_proj : str or Proj, optional
        Source projection. Default is 'epsg:4326' (WGS84).
    dst_proj : str or Proj, optional
        Destination projection. Required.
        
    Returns
    -------
    tuple
        (x, y) tuple with transformed coordinates.
    """
    check_pyproj()
    
    if dst_proj is None:
        raise ValueError("Destination projection (dst_proj) is required")
    
    # Create transformer
    transformer = Transformer.from_proj(src_proj, dst_proj, always_xy=True)
    
    # Transform coordinates
    x, y = transformer.transform(lons, lats)
    
    return x, y


def get_proj4_string(projection):
    """
    Get the PROJ.4 string for a projection.
    
    Parameters
    ----------
    projection : Proj
        Projection object.
        
    Returns
    -------
    str
        PROJ.4 string.
    """
    check_pyproj()
    
    return projection.srs