"""Module for creating base maps with borders for visualization."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from eoftoolkit.core.exceptions import VisualizationError


def create_basemap(lats, lons, projection='merc', padding=1, resolution='i', **kwargs):
    """
    Create a Basemap object for visualization.
    
    Parameters
    ----------
    lats : ndarray
        2D grid of latitude values.
    lons : ndarray
        2D grid of longitude values.
    projection : str, optional
        Map projection to use. Default is 'merc' (Mercator).
    padding : float, optional
        Padding to add around the data extent in degrees. Default is 1.
    resolution : str, optional
        Resolution of map boundary database. Default is 'i' (intermediate).
    **kwargs : dict
        Additional parameters to pass to Basemap constructor.
        For LCC projection, could include lat_0, lon_0, lat_1, lat_2.
        
    Returns
    -------
    Basemap
        Basemap object for plotting.
    """
    try:
        # Determine map boundaries
        if len(lats.shape) > 1:
            # If 2D grids are provided
            lat_min = np.nanmin(lats) - padding
            lat_max = np.nanmax(lats) + padding
            lon_min = np.nanmin(lons) - padding
            lon_max = np.nanmax(lons) + padding
        else:
            # If 1D arrays are provided
            lat_min = np.min(lats) - padding
            lat_max = np.max(lats) + padding
            lon_min = np.min(lons) - padding
            lon_max = np.max(lons) + padding
        
        # Base parameters for the map
        map_params = {
            'projection': projection,
            'resolution': resolution,
            'llcrnrlat': lat_min,
            'urcrnrlat': lat_max,
            'llcrnrlon': lon_min,
            'urcrnrlon': lon_max
        }
        
        # Update with any additional parameters
        map_params.update(kwargs)
        
        # Create the Basemap object
        m = Basemap(**map_params)
        
        return m
    
    except Exception as e:
        raise VisualizationError(f"Error creating basemap: {str(e)}")


def add_map_features(m, draw_coastlines=True, draw_countries=True, 
                  draw_states=False, draw_parallels=True, draw_meridians=True,
                  parallel_step=5, meridian_step=10):
    """
    Add features to a Basemap object.
    
    Parameters
    ----------
    m : Basemap
        Basemap object to add features to.
    draw_coastlines : bool, optional
        Whether to draw coastlines. Default is True.
    draw_countries : bool, optional
        Whether to draw country borders. Default is True.
    draw_states : bool, optional
        Whether to draw state/province borders. Default is False.
    draw_parallels : bool, optional
        Whether to draw parallels. Default is True.
    draw_meridians : bool, optional
        Whether to draw meridians. Default is True.
    parallel_step : int, optional
        Step size for parallels in degrees. Default is 5.
    meridian_step : int, optional
        Step size for meridians in degrees. Default is 10.
        
    Returns
    -------
    Basemap
        Basemap object with added features.
    """
    try:
        # Add map features
        if draw_coastlines:
            m.drawcoastlines()
        
        if draw_countries:
            m.drawcountries()
        
        if draw_states:
            m.drawstates()
        
        if draw_parallels:
            lat_min = m.llcrnrlat
            lat_max = m.urcrnrlat
            parallels = np.arange(
                np.floor(lat_min / parallel_step) * parallel_step,
                np.ceil(lat_max / parallel_step) * parallel_step,
                parallel_step
            )
            m.drawparallels(parallels, labels=[1, 0, 0, 0])
        
        if draw_meridians:
            lon_min = m.llcrnrlon
            lon_max = m.urcrnrlon
            meridians = np.arange(
                np.floor(lon_min / meridian_step) * meridian_step,
                np.ceil(lon_max / meridian_step) * meridian_step,
                meridian_step
            )
            m.drawmeridians(meridians, labels=[0, 0, 0, 1])
        
        return m
    
    except Exception as e:
        raise VisualizationError(f"Error adding map features: {str(e)}")