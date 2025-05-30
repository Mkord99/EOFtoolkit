"""Module for visualizing reconstructed data."""

import numpy as np
import matplotlib.pyplot as plt
from eoftoolkit.visualization.spatial import plot_spatial_field
from eoftoolkit.processor.reshaper import reshape_to_spatial_grid
from eoftoolkit.core.exceptions import VisualizationError


def plot_reconstruction(reconstruction, id_matrix, lats, lons, 
                      timestamp_index=0, title=None, cmap='RdBu_r',
                      projection='merc', contour_levels=21, 
                      show_colorbar=True, colorbar_label=None,
                      fig=None, ax=None, save_path=None, **projection_params):
    """
    Plot a reconstruction at a specific timestamp.
    
    Parameters
    ----------
    reconstruction : ndarray
        2D array with reconstructed data (rows=timestamps, columns=spatial locations).
    id_matrix : ndarray
        ID matrix used for reshaping.
    lats : ndarray
        2D grid of latitude values.
    lons : ndarray
        2D grid of longitude values.
    timestamp_index : int, optional
        Index of the timestamp to plot. Default is 0.
    title : str, optional
        Title for the plot. If None, a default title is generated.
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
    **projection_params : dict
        Additional parameters to pass to the map projection.
        
    Returns
    -------
    tuple
        (fig, ax) tuple with the figure and axes objects.
    """
    try:
        # Extract the reconstruction for the specified timestamp
        if len(reconstruction.shape) > 1:
            recon_data = reconstruction[timestamp_index, :]
        else:
            recon_data = reconstruction
        
        # Set default title if not provided
        if title is None:
            title = f"Reconstructed Data (Timestamp {timestamp_index})"
        
        # Plot using the spatial plotting function
        fig, ax = plot_spatial_field(
            data=recon_data,
            lats=lats,
            lons=lons,
            id_matrix=id_matrix,
            title=title,
            cmap=cmap,
            projection=projection,
            contour_levels=contour_levels,
            show_colorbar=show_colorbar,
            colorbar_label=colorbar_label,
            fig=fig,
            ax=ax,
            save_path=save_path,
            **projection_params
        )
        
        return fig, ax
    
    except Exception as e:
        raise VisualizationError(f"Error plotting reconstruction: {str(e)}")


def plot_reconstruction_sequence(reconstruction, id_matrix, lats, lons, 
                               timestamps=None, n_plots=4, title_prefix=None,
                               cmap='RdBu_r', projection='merc', 
                               contour_levels=21, figsize=(18, 12),
                               save_path=None, **projection_params):
    """
    Plot a sequence of reconstructions at different timestamps.
    
    Parameters
    ----------
    reconstruction : ndarray
        2D array with reconstructed data (rows=timestamps, columns=spatial locations).
    id_matrix : ndarray
        ID matrix used for reshaping.
    lats : ndarray
        2D grid of latitude values.
    lons : ndarray
        2D grid of longitude values.
    timestamps : list, optional
        List of timestamp indices to plot. If None, evenly spaced timestamps are selected.
    n_plots : int, optional
        Number of plots if timestamps is None. Default is 4.
    title_prefix : str, optional
        Prefix for the titles. If None, a default prefix is used.
    cmap : str, optional
        Colormap to use. Default is 'RdBu_r'.
    projection : str, optional
        Map projection. Default is 'merc'.
    contour_levels : int, optional
        Number of contour levels. Default is 21.
    figsize : tuple, optional
        Figure size. Default is (18, 12).
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
        # Determine the timestamps to plot
        if timestamps is None:
            n_timestamps = reconstruction.shape[0]
            timestamps = np.linspace(0, n_timestamps - 1, n_plots, dtype=int)
        else:
            n_plots = len(timestamps)
        
        # Calculate grid dimensions
        n_cols = min(2, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))
        
        # Create figure and axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Flatten axes if it's a 2D array
        if n_plots > 1:
            if n_rows > 1 and n_cols > 1:
                axes = axes.flatten()
            elif n_rows == 1:
                axes = [axes[i] for i in range(n_cols)]
            elif n_cols == 1:
                axes = [axes[i] for i in range(n_rows)]
        else:
            axes = [axes]
        
        # Set default title prefix if not provided
        if title_prefix is None:
            title_prefix = "Reconstructed Data"
        
        # Create global colormap limits for consistent coloring
        all_data = []
        for idx in timestamps:
            reshaped = reshape_to_spatial_grid(reconstruction[idx, :], id_matrix)
            all_data.append(reshaped)
        
        all_data = np.vstack([data.flatten() for data in all_data])
        vmin = np.nanmin(all_data)
        vmax = np.nanmax(all_data)
        clevs = np.linspace(vmin, vmax, contour_levels)
        
        # Plot each timestamp
        for i, idx in enumerate(timestamps):
            if i >= len(axes):
                break
                
            recon_data = reconstruction[idx, :]
            title = f"{title_prefix} (Timestamp {idx})"
            
            # Reshape the data
            reshaped_data = reshape_to_spatial_grid(recon_data, id_matrix)
            masked_data = np.ma.masked_invalid(reshaped_data)
            
            ax = axes[i]
            
            # Create basemap
            from eoftoolkit.visualization.base_maps import create_basemap, add_map_features
            m = create_basemap(lats, lons, projection=projection, **projection_params)
            add_map_features(m)
            
            # Create coordinate grids for plotting
            lon_grid, lat_grid = np.meshgrid(
                np.linspace(m.llcrnrlon, m.urcrnrlon, lons.shape[1]),
                np.linspace(m.llcrnrlat, m.urcrnrlat, lats.shape[0])
            )
            x, y = m(lon_grid, lat_grid)
            
            # Plot contour
            cs = m.contourf(x, y, masked_data, levels=clevs, cmap=cmap, extend="both", ax=ax)
            
            # Add colorbar
            cb = m.colorbar(cs, location='right', pad="5%", ax=ax)
            
            # Set title
            ax.set_title(title, fontsize=12)
            
            # Add labels
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        
        # Hide any empty subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes
    
    except Exception as e:
        raise VisualizationError(f"Error plotting reconstruction sequence: {str(e)}")