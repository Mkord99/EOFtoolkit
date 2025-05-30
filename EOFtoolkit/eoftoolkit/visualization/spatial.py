"""Module for general spatial plotting utilities."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from eoftoolkit.visualization.base_maps import create_basemap, add_map_features
from eoftoolkit.processor.reshaper import reshape_to_spatial_grid
from eoftoolkit.core.exceptions import VisualizationError

# Import the improved plotting functions if available
try:
    from eoftoolkit.visualization.plotting_improvements import (
        create_figure_with_fixed_layout,
        create_comparison_layout,
        add_colorbar
    )
    IMPROVED_PLOTTING = True
except ImportError:
    IMPROVED_PLOTTING = False


def plot_spatial_field(data, lats, lons, id_matrix=None, title=None, 
                     cmap='RdBu_r', projection='merc', contour_levels=21, 
                     show_colorbar=True, colorbar_label=None,
                     fig=None, ax=None, save_path=None, figsize=(10, 8),
                     **projection_params):
    """
    Plot a spatial field on a map.
    
    Parameters
    ----------
    data : ndarray
        Data to plot. Can be 2D (already in spatial grid) or 1D (needs reshaping).
    lats : ndarray
        2D grid of latitude values.
    lons : ndarray
        2D grid of longitude values.
    id_matrix : ndarray, optional
        ID matrix used for reshaping if data is 1D.
    title : str, optional
        Title for the plot.
    cmap : str, optional
        Colormap to use. Default is 'RdBu_r'.
    projection : str, optional
        Map projection. Default is 'merc'.
    contour_levels : int, optional
        Number of contour levels. Default is 21.
    show_colorbar : bool, optional
        Whether to show a colorbar. Default is True.
    colorbar_label : str, optional
        Label for the colorbar.
    fig : Figure, optional
        Matplotlib Figure object to plot on. If None, a new figure is created.
    ax : Axes, optional
        Matplotlib Axes object to plot on. If None, a new axes is created.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    figsize : tuple, optional
        Figure size in inches. Default is (10, 8).
    **projection_params : dict
        Additional parameters to pass to the map projection.
        
    Returns
    -------
    tuple
        (fig, ax) tuple with the figure and axes objects.
    """
    try:
        # Reshape data if it's 1D and id_matrix is provided
        if len(data.shape) == 1 and id_matrix is not None:
            data = reshape_to_spatial_grid(data, id_matrix)
        
        # Create a masked array
        masked_data = np.ma.masked_invalid(data)
        
        # Create figure and axes if not provided
        if fig is None or ax is None:
            if IMPROVED_PLOTTING:
                fig, ax = create_figure_with_fixed_layout(figsize=figsize)
            else:
                fig, ax = plt.subplots(figsize=figsize)
        
        # Create basemap
        m = create_basemap(lats, lons, projection=projection, **projection_params)
        
        # Add map features
        add_map_features(m)
        
        # Create coordinate grids for plotting
        if len(lons.shape) == 1 and len(lats.shape) == 1:
            # If 1D arrays are provided
            lon_grid, lat_grid = np.meshgrid(lons, lats)
        else:
            # If 2D grids are provided
            lon_grid, lat_grid = lons, lats
        
        x, y = m(lon_grid, lat_grid)
        
        # Create contour levels
        try:
            vmin = np.nanmin(masked_data)
            vmax = np.nanmax(masked_data)
            
            # Handle case where min and max are equal
            if vmin == vmax:
                vmin = vmin - 0.1
                vmax = vmax + 0.1
            
            clevs = np.linspace(vmin, vmax, contour_levels)
        except ValueError:
            # If all data is NaN, use default levels
            clevs = np.linspace(-1, 1, contour_levels)
        
        # Plot contour
        cs = m.contourf(x, y, masked_data, levels=clevs, cmap=cmap, extend="both")
        
        # Add colorbar
        if show_colorbar:
            if IMPROVED_PLOTTING:
                cb = add_colorbar(fig, cs, ax=ax, label=colorbar_label, pad=0.05)
            else:
                cb = m.colorbar(cs, location='right', pad="5%")
                if colorbar_label:
                    cb.set_label(colorbar_label)
        
        # Set title
        if title:
            ax.set_title(title, fontsize=12)
        
        # Add labels
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        # Ensure proper layout
        if hasattr(fig, 'set_constrained_layout'):
            fig.set_constrained_layout(True)
        else:
            plt.tight_layout()
        
        # Save figure if requested
        if save_path is not None:
            if IMPROVED_PLOTTING:
                from eoftoolkit.visualization.plotting_improvements import save_figure_properly
                save_figure_properly(fig, save_path)
            else:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    except Exception as e:
        raise VisualizationError(f"Error plotting spatial field: {str(e)}")


def plot_reconstruction_comparison(original, reconstructed, id_matrix, lats, lons,
                                 timestamp_index=None, title_prefix=None, projection='merc',
                                 cmap='RdBu_r', diff_cmap='RdBu_r', figsize=(18, 6),
                                 save_path=None, **projection_params):
    """
    Plot a comparison between original and reconstructed data with difference.
    
    Parameters
    ----------
    original : ndarray
        Original data (1D).
    reconstructed : ndarray
        Reconstructed data (1D).
    id_matrix : ndarray
        ID matrix used for reshaping.
    lats : ndarray
        2D grid of latitude values.
    lons : ndarray
        2D grid of longitude values.
    timestamp_index : int, optional
        Index of the timestamp to use in the title.
    title_prefix : str, optional
        Prefix for the subplot titles.
    projection : str, optional
        Map projection. Default is 'merc'.
    cmap : str, optional
        Colormap for data. Default is 'RdBu_r'.
    diff_cmap : str, optional
        Colormap for difference. Default is 'RdBu_r'.
    figsize : tuple, optional
        Figure size. Default is (18, 6).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    **projection_params : dict
        Additional parameters to pass to the map projection.
        
    Returns
    -------
    tuple
        (fig, axes) tuple with the figure and axes objects.
    """
    try:
        # Create figure and axes with improved layout
        if IMPROVED_PLOTTING:
            fig, (ax1, ax2, ax3) = create_comparison_layout(figsize=figsize)
        else:
            # Create figure and axes
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(1, 3, width_ratios=[1, 1, 1])
            
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
        
        # Reshape data
        reshaped_original = reshape_to_spatial_grid(original, id_matrix)
        reshaped_reconstructed = reshape_to_spatial_grid(reconstructed, id_matrix)
        reshaped_difference = reshaped_reconstructed - reshaped_original
        
        # Create masked arrays
        masked_original = np.ma.masked_invalid(reshaped_original)
        masked_reconstructed = np.ma.masked_invalid(reshaped_reconstructed)
        masked_difference = np.ma.masked_invalid(reshaped_difference)
        
        # Create titles
        if title_prefix is None:
            title_prefix = ""
        else:
            title_prefix = f"{title_prefix} - "
        
        if timestamp_index is not None:
            title_original = f"{title_prefix}Original (Timestamp {timestamp_index})"
            title_reconstructed = f"{title_prefix}Reconstructed (Timestamp {timestamp_index})"
            title_difference = f"{title_prefix}Difference (Timestamp {timestamp_index})"
        else:
            title_original = f"{title_prefix}Original"
            title_reconstructed = f"{title_prefix}Reconstructed"
            title_difference = f"{title_prefix}Difference"
        
        # Plot original
        m1 = create_basemap(lats, lons, projection=projection, **projection_params)
        add_map_features(m1)
        lon_grid, lat_grid = np.meshgrid(
            np.linspace(m1.llcrnrlon, m1.urcrnrlon, lons.shape[1]),
            np.linspace(m1.llcrnrlat, m1.urcrnrlat, lats.shape[0])
        )
        x, y = m1(lon_grid, lat_grid)
        vmin = np.nanmin(masked_original)
        vmax = np.nanmax(masked_original)
        clevs = np.linspace(vmin, vmax, 21)
        cs1 = m1.contourf(x, y, masked_original, levels=clevs, cmap=cmap, extend="both", ax=ax1)
        
        if IMPROVED_PLOTTING:
            cb1 = add_colorbar(fig, cs1, ax=ax1, pad=0.05)
        else:
            cb1 = m1.colorbar(cs1, location='right', pad="5%", ax=ax1)
        
        ax1.set_title(title_original)
        
        # Plot reconstructed
        m2 = create_basemap(lats, lons, projection=projection, **projection_params)
        add_map_features(m2)
        cs2 = m2.contourf(x, y, masked_reconstructed, levels=clevs, cmap=cmap, extend="both", ax=ax2)
        
        if IMPROVED_PLOTTING:
            cb2 = add_colorbar(fig, cs2, ax=ax2, pad=0.05)
        else:
            cb2 = m2.colorbar(cs2, location='right', pad="5%", ax=ax2)
            
        ax2.set_title(title_reconstructed)
        
        # Plot difference
        m3 = create_basemap(lats, lons, projection=projection, **projection_params)
        add_map_features(m3)
        vmin_diff = np.nanmin(masked_difference)
        vmax_diff = np.nanmax(masked_difference)
        # Make divergent colormap centered at 0
        abs_max = max(abs(vmin_diff), abs(vmax_diff))
        clevs_diff = np.linspace(-abs_max, abs_max, 21)
        cs3 = m3.contourf(x, y, masked_difference, levels=clevs_diff, cmap=diff_cmap, extend="both", ax=ax3)
        
        if IMPROVED_PLOTTING:
            cb3 = add_colorbar(fig, cs3, ax=ax3, pad=0.05)
        else:
            cb3 = m3.colorbar(cs3, location='right', pad="5%", ax=ax3)
            
        ax3.set_title(title_difference)
        
        # Ensure proper layout
        if hasattr(fig, 'set_constrained_layout'):
            fig.set_constrained_layout(True)
        else:
            plt.tight_layout()
        
        # Save figure if requested
        if save_path is not None:
            if IMPROVED_PLOTTING:
                from eoftoolkit.visualization.plotting_improvements import save_figure_properly
                save_figure_properly(fig, save_path)
            else:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, (ax1, ax2, ax3)
    
    except Exception as e:
        raise VisualizationError(f"Error plotting reconstruction comparison: {str(e)}")