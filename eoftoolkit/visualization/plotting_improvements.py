"""Module for visualizing EOFs on maps."""

import numpy as np
import matplotlib.pyplot as plt
from eoftoolkit.visualization.base_maps import create_basemap, add_map_features
from eoftoolkit.processor.reshaper import reshape_to_spatial_grid
from eoftoolkit.core.exceptions import VisualizationError

# Import the improved plotting functions if available
try:
    from eoftoolkit.visualization.plotting_improvements import (
        create_figure_with_fixed_layout,
        add_colorbar
    )
    IMPROVED_PLOTTING = True
except ImportError:
    IMPROVED_PLOTTING = False


def plot_eof(eof, id_matrix, lats, lons, mode_number=1, title=None, 
             cmap='RdBu_r', projection='merc', contour_levels=21, 
             show_colorbar=True, fig=None, ax=None, save_path=None, 
             **projection_params):
    """
    Plot an EOF pattern on a map.
    
    Parameters
    ----------
    eof : ndarray
        1D array containing EOF values.
    id_matrix : ndarray
        ID matrix used for reshaping.
    lats : ndarray
        2D grid of latitude values.
    lons : ndarray
        2D grid of longitude values.
    mode_number : int, optional
        Mode number for the title. Default is 1.
    title : str, optional
        Custom title for the plot. If None, a default title is generated.
    cmap : str, optional
        Colormap to use. Default is 'RdBu_r'.
    projection : str, optional
        Map projection. Default is 'merc'.
    contour_levels : int, optional
        Number of contour levels. Default is 21.
    show_colorbar : bool, optional
        Whether to show a colorbar. Default is True.
    fig : Figure, optional
        Matplotlib Figure object to plot on. If None, a new figure is created.
    ax : Axes, optional
        Matplotlib Axes object to plot on. If None, a new axes is created.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    **projection_params : dict
        Additional parameters to pass to create_basemap for the projection.
        For 'lcc' projection, this could include 'lat_1', 'lat_2', 'lat_0', 'lon_0', etc.
        
    Returns
    -------
    tuple
        (fig, ax) tuple with the figure and axes objects.
    """
    try:
        # Reshape the EOF to a 2D grid
        reshaped_eof = reshape_to_spatial_grid(eof, id_matrix)
        
        # Create a masked array
        masked_eof = np.ma.masked_invalid(reshaped_eof)
        
        # Create figure and axes if not provided
        if fig is None or ax is None:
            if IMPROVED_PLOTTING:
                fig, ax = create_figure_with_fixed_layout(figsize=(10, 8))
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
        
        # Check if constrained layout is enabled
        constrained_layout = hasattr(fig, 'get_constrained_layout') and fig.get_constrained_layout()
        
        # Create basemap
        m = create_basemap(lats, lons, projection=projection, **projection_params)
        
        # Add map features
        add_map_features(m)
        
        # Create coordinate grids for plotting
        lon_grid, lat_grid = np.meshgrid(
            np.linspace(m.llcrnrlon, m.urcrnrlon, lons.shape[1]),
            np.linspace(m.llcrnrlat, m.urcrnrlat, lats.shape[0])
        )
        x, y = m(lon_grid, lat_grid)
        
        # Create contour levels
        clevs = np.linspace(masked_eof.min(), masked_eof.max(), contour_levels)
        
        # Plot contour
        cs = m.contourf(x, y, masked_eof, levels=clevs, cmap=cmap, extend="both")
        
        # Add colorbar
        if show_colorbar:
            if IMPROVED_PLOTTING:
                cb = add_colorbar(fig, cs, ax=ax, pad=0.05)
            else:
                cb = m.colorbar(cs, location='right', pad="5%")
                cb.set_label("EOF Amplitude")
        
        # Set title
        if title is None:
            title = f"EOF {mode_number}"
        ax.set_title(title, fontsize=12)
        
        # Add labels
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
        # Only do tight_layout if constrained_layout is not enabled
        # This avoids the conflict between layout engines
        if not constrained_layout:
            try:
                # Skip tight_layout if a colorbar was created (to avoid the layout conflict)
                if not show_colorbar:
                    plt.tight_layout()
            except Exception:
                # If tight_layout fails, just continue
                pass
        
        # Save figure if requested
        if save_path is not None:
            if IMPROVED_PLOTTING:
                from eoftoolkit.visualization.plotting_improvements import save_figure_properly
                save_figure_properly(fig, save_path)
            else:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    except Exception as e:
        raise VisualizationError(f"Error plotting EOF: {str(e)}")


def plot_multiple_eofs(eofs, id_matrix, lats, lons, mode_numbers=None,
                    titles=None, cmap='RdBu_r', projection='merc',
                    contour_levels=21, show_colorbar=True, 
                    figsize=(15, 10), save_path=None, **projection_params):
    """
    Plot multiple EOFs on a grid of maps.
    
    Parameters
    ----------
    eofs : ndarray
        2D array containing multiple EOFs (rows=modes, columns=spatial locations).
    id_matrix : ndarray
        ID matrix used for reshaping.
    lats : ndarray
        2D grid of latitude values.
    lons : ndarray
        2D grid of longitude values.
    mode_numbers : list, optional
        List of mode numbers. If None, uses [1, 2, ..., n].
    titles : list, optional
        List of titles for each subplot. If None, default titles are generated.
    cmap : str, optional
        Colormap to use. Default is 'RdBu_r'.
    projection : str, optional
        Map projection. Default is 'merc'.
    contour_levels : int, optional
        Number of contour levels. Default is 21.
    show_colorbar : bool, optional
        Whether to show colorbars. Default is True.
    figsize : tuple, optional
        Figure size. Default is (15, 10).
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
    **projection_params : dict
        Additional parameters to pass to create_basemap for the projection.
        
    Returns
    -------
    tuple
        (fig, axes) tuple with the figure and axes objects.
    """
    try:
        n_eofs = eofs.shape[0]
        
        # Calculate grid dimensions
        n_cols = min(3, n_eofs)
        n_rows = int(np.ceil(n_eofs / n_cols))
        
        # Create figure and axes
        if IMPROVED_PLOTTING:
            fig, axes = create_figure_with_fixed_layout(nrows=n_rows, ncols=n_cols, figsize=figsize)
            # Flatten for easier indexing
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()
            elif n_rows > 1 or n_cols > 1:
                axes = np.array(axes).flatten()
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            # Flatten axes if it's a 2D array
            if n_eofs > 1:
                if n_rows > 1 and n_cols > 1:
                    axes = axes.flatten()
                elif n_rows == 1:
                    axes = [axes[i] for i in range(n_cols)]
                elif n_cols == 1:
                    axes = [axes[i] for i in range(n_rows)]
            else:
                axes = [axes]
        
        # Create default mode numbers if not provided
        if mode_numbers is None:
            mode_numbers = list(range(1, n_eofs + 1))
        
        # Create default titles if not provided
        if titles is None:
            titles = [f"EOF {mode}" for mode in mode_numbers]
        
        # Plot each EOF
        for i in range(n_eofs):
            if i >= len(axes):
                break
                
            eof = eofs[i, :]
            ax = axes[i]
            
            # Create basemap for this subplot
            m = create_basemap(lats, lons, projection=projection, **projection_params)
            
            # Add map features
            add_map_features(m)
            
            # Reshape the EOF to a 2D grid
            reshaped_eof = reshape_to_spatial_grid(eof, id_matrix)
            
            # Create a masked array
            masked_eof = np.ma.masked_invalid(reshaped_eof)
            
            # Create coordinate grids for plotting
            lon_grid, lat_grid = np.meshgrid(
                np.linspace(m.llcrnrlon, m.urcrnrlon, lons.shape[1]),
                np.linspace(m.llcrnrlat, m.urcrnrlat, lats.shape[0])
            )
            x, y = m(lon_grid, lat_grid)
            
            # Create contour levels
            clevs = np.linspace(masked_eof.min(), masked_eof.max(), contour_levels)
            
            # Plot contour
            cs = m.contourf(x, y, masked_eof, levels=clevs, cmap=cmap, extend="both", ax=ax)
            
            # Add colorbar
            if show_colorbar:
                if IMPROVED_PLOTTING:
                    cb = add_colorbar(fig, cs, ax=ax, pad=0.05)
                else:
                    cb = m.colorbar(cs, location='right', pad="5%", ax=ax)
                    cb.set_label("EOF Amplitude")
            
            # Set title
            ax.set_title(titles[i], fontsize=12)
            
            # Add labels
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        
        # Hide any empty subplots
        for i in range(n_eofs, len(axes)):
            axes[i].axis('off')
        
        # Only do tight_layout if constrained_layout is not enabled
        if not hasattr(fig, 'get_constrained_layout') or not fig.get_constrained_layout():
            try:
                # Skip tight_layout if colorbars were created
                if not show_colorbar:
                    plt.tight_layout()
            except Exception:
                # If tight_layout fails, just continue
                pass
        
        # Save figure if requested
        if save_path is not None:
            if IMPROVED_PLOTTING:
                from eoftoolkit.visualization.plotting_improvements import save_figure_properly
                save_figure_properly(fig, save_path)
            else:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes
    
    except Exception as e:
        raise VisualizationError(f"Error plotting multiple EOFs: {str(e)}")