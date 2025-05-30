"""Module for visualizing error metrics and differences."""

import numpy as np
import matplotlib.pyplot as plt
from eoftoolkit.visualization.spatial import plot_spatial_field
from eoftoolkit.processor.reshaper import reshape_to_spatial_grid
from eoftoolkit.core.exceptions import VisualizationError


def plot_reconstruction_error(error_metrics, metric_name='rmse', title=None,
                            xlabel='Number of Modes', ylabel=None,
                            color='blue', marker='o', linestyle='-',
                            figsize=(10, 6), grid=True,
                            fig=None, ax=None, save_path=None):
    """
    Plot reconstruction error metrics as a function of number of modes.
    
    Parameters
    ----------
    error_metrics : dict
        Dictionary with mode counts as keys and error metrics as values.
    metric_name : str, optional
        Name of the metric to plot. Default is 'rmse'.
    title : str, optional
        Title for the plot. If None, a default title is generated.
    xlabel : str, optional
        Label for the x-axis. Default is 'Number of Modes'.
    ylabel : str, optional
        Label for the y-axis. If None, uses the metric name.
    color : str, optional
        Line color. Default is 'blue'.
    marker : str, optional
        Marker style. Default is 'o'.
    linestyle : str, optional
        Line style. Default is '-'.
    figsize : tuple, optional
        Figure size. Default is (10, 6).
    grid : bool, optional
        Whether to show grid. Default is True.
    fig : Figure, optional
        Matplotlib Figure object to plot on. If None, a new figure is created.
    ax : Axes, optional
        Matplotlib Axes object to plot on. If None, a new axes is created.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns
    -------
    tuple
        (fig, ax) tuple with the figure and axes objects.
    """
    try:
        # Create figure and axes if not provided
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Extract x and y values
        mode_counts = sorted(error_metrics.keys())
        metric_values = [error_metrics[count][metric_name] for count in mode_counts]
        
        # Plot error metrics
        ax.plot(mode_counts, metric_values, color=color, marker=marker, 
               linestyle=linestyle, label=metric_name.upper())
        
        # Set title
        if title is None:
            title = f"Reconstruction Error ({metric_name.upper()}) vs. Number of Modes"
        ax.set_title(title, fontsize=12)
        
        # Set labels
        ax.set_xlabel(xlabel, fontsize=10)
        if ylabel is None:
            ylabel = metric_name.upper()
        ax.set_ylabel(ylabel, fontsize=10)
        
        # Set x-ticks to integers
        ax.set_xticks(mode_counts)
        
        # Add grid
        if grid:
            ax.grid(True, linestyle="--", alpha=0.7)
        
        # Add legend
        ax.legend(fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    except Exception as e:
        raise VisualizationError(f"Error plotting reconstruction error: {str(e)}")


def plot_multiple_error_metrics(error_metrics, metrics=None, title=None,
                             xlabel='Number of Modes', colors=None,
                             markers=None, figsize=(12, 8), grid=True,
                             fig=None, ax=None, save_path=None):
    """
    Plot multiple reconstruction error metrics as functions of number of modes.
    
    Parameters
    ----------
    error_metrics : dict
        Dictionary with mode counts as keys and error metrics as values.
    metrics : list, optional
        List of metric names to plot. If None, plots rmse, mae, and r2.
    title : str, optional
        Title for the plot. If None, a default title is generated.
    xlabel : str, optional
        Label for the x-axis. Default is 'Number of Modes'.
    colors : list, optional
        List of colors for each metric. If None, uses default colors.
    markers : list, optional
        List of markers for each metric. If None, uses default markers.
    figsize : tuple, optional
        Figure size. Default is (12, 8).
    grid : bool, optional
        Whether to show grid. Default is True.
    fig : Figure, optional
        Matplotlib Figure object to plot on. If None, a new figure is created.
    ax : Axes, optional
        Matplotlib Axes object to plot on. If None, a new axes is created.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns
    -------
    tuple
        (fig, ax) tuple with the figure and axes objects.
    """
    try:
        # Create figure and axes if not provided
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Set default metrics if not provided
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2']
        
        # Set default colors if not provided
        if colors is None:
            colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        # Set default markers if not provided
        if markers is None:
            markers = ['o', 's', '^', 'D', 'x']
        
        # Extract x values
        mode_counts = sorted(error_metrics.keys())
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # Extract y values
            try:
                metric_values = [error_metrics[count][metric] for count in mode_counts]
                
                # For r2 and explained_variance, higher is better, so plot on the secondary y-axis
                if metric in ['r2', 'explained_variance']:
                    ax2 = ax.twinx()
                    ax2.plot(mode_counts, metric_values, color=color, marker=marker,
                           linestyle='-', label=metric.upper())
                    ax2.set_ylabel(f"{metric.upper()} (higher is better)", color=color, fontsize=10)
                    ax2.tick_params(axis='y', labelcolor=color)
                    ax2.legend(loc='upper right', fontsize=10)
                else:
                    # For other metrics, lower is better
                    ax.plot(mode_counts, metric_values, color=color, marker=marker,
                          linestyle='-', label=metric.upper())
            except KeyError:
                continue
        
        # Set title
        if title is None:
            title = "Reconstruction Error Metrics vs. Number of Modes"
        ax.set_title(title, fontsize=12)
        
        # Set labels for primary axis
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Error (lower is better)", fontsize=10)
        
        # Set x-ticks to integers
        ax.set_xticks(mode_counts)
        
        # Add grid
        if grid:
            ax.grid(True, linestyle="--", alpha=0.7)
        
        # Add legend for primary axis
        ax.legend(loc='upper left', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    except Exception as e:
        raise VisualizationError(f"Error plotting multiple error metrics: {str(e)}")


def plot_error_spatial(original, reconstruction, id_matrix, lats, lons,
                     timestamp_index=0, error_type='difference',
                     title=None, cmap='RdBu_r', projection='merc',
                     contour_levels=21, show_colorbar=True,
                     colorbar_label=None, fig=None, ax=None,
                     save_path=None):
    """
    Plot spatial pattern of reconstruction error.
    
    Parameters
    ----------
    original : ndarray
        Original data (can be 2D with rows=timestamps, or 1D for a single timestamp).
    reconstruction : ndarray
        Reconstructed data (can be 2D with rows=timestamps, or 1D for a single timestamp).
    id_matrix : ndarray
        ID matrix used for reshaping.
    lats : ndarray
        2D grid of latitude values.
    lons : ndarray
        2D grid of longitude values.
    timestamp_index : int, optional
        Index of the timestamp to plot. Default is 0.
    error_type : str, optional
        Type of error to plot. Can be 'difference', 'squared_difference',
        'absolute_difference', or 'percent_difference'. Default is 'difference'.
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
        
    Returns
    -------
    tuple
        (fig, ax) tuple with the figure and axes objects.
    """
    try:
        # Extract the data for the specified timestamp
        if len(original.shape) > 1:
            orig_data = original[timestamp_index, :]
        else:
            orig_data = original
            
        if len(reconstruction.shape) > 1:
            recon_data = reconstruction[timestamp_index, :]
        else:
            recon_data = reconstruction
        
        # Calculate error
        if error_type == 'difference':
            error_data = recon_data - orig_data
            if colorbar_label is None:
                colorbar_label = "Difference"
        elif error_type == 'squared_difference':
            error_data = (recon_data - orig_data) ** 2
            if colorbar_label is None:
                colorbar_label = "Squared Difference"
        elif error_type == 'absolute_difference':
            error_data = np.abs(recon_data - orig_data)
            if colorbar_label is None:
                colorbar_label = "Absolute Difference"
        elif error_type == 'percent_difference':
            # Avoid division by zero
            mask = (orig_data != 0)
            error_data = np.zeros_like(orig_data)
            error_data[mask] = 100 * (recon_data[mask] - orig_data[mask]) / orig_data[mask]
            if colorbar_label is None:
                colorbar_label = "Percent Difference (%)"
        else:
            raise ValueError(f"Invalid error_type: {error_type}")
        
        # Set default title if not provided
        if title is None:
            title = f"Reconstruction Error ({error_type.replace('_', ' ').title()}, Timestamp {timestamp_index})"
        
        # Plot using the spatial plotting function
        fig, ax = plot_spatial_field(
            data=error_data,
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
            save_path=save_path
        )
        
        return fig, ax
    
    except Exception as e:
        raise VisualizationError(f"Error plotting error spatial pattern: {str(e)}")