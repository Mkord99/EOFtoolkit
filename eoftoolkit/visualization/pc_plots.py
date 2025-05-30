"""Module for visualizing Principal Components (PCs) as time series."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from eoftoolkit.core.exceptions import VisualizationError


def plot_pc(pc, dates=None, mode_number=1, title=None, 
           color='blue', marker='.', linestyle='-', linewidth=1, 
           figsize=(12, 6), grid=True, show_markers=True,
           date_format=None, fig=None, ax=None, save_path=None):
    """
    Plot a Principal Component (PC) time series.
    
    Parameters
    ----------
    pc : ndarray
        1D array containing PC values.
    dates : list or ndarray, optional
        Dates or time values corresponding to PC values.
        If None, uses index values.
    mode_number : int, optional
        Mode number for the title. Default is 1.
    title : str, optional
        Custom title for the plot. If None, a default title is generated.
    color : str, optional
        Line color. Default is 'blue'.
    marker : str, optional
        Marker style. Default is '.'.
    linestyle : str, optional
        Line style. Default is '-'.
    linewidth : float, optional
        Line width. Default is 1.
    figsize : tuple, optional
        Figure size. Default is (12, 6).
    grid : bool, optional
        Whether to show grid. Default is True.
    show_markers : bool, optional
        Whether to show markers. Default is True.
    date_format : str, optional
        Format string for parsing dates (e.g., '%Y%m' for '202301'). 
        Default is None (auto-detect).
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
        
        # Flatten PC if needed
        if len(pc.shape) > 1:
            pc = pc.flatten()
        
        # Create x values
        if dates is None:
            x = np.arange(len(pc))
        else:
            x = dates
            
            # Convert string dates to datetime if needed
            if isinstance(x[0], str):
                try:
                    if date_format:
                        # Use specified format
                        x = pd.to_datetime(x, format=date_format)
                    else:
                        # Try auto-detection
                        x = pd.to_datetime(x)
                except:
                    # If conversion fails, use indices
                    x = np.arange(len(pc))
        
        # Plot PC
        ax.plot(x, pc, color=color, linestyle=linestyle, linewidth=linewidth, label=f"PC {mode_number}")
        
        # Add markers if requested
        if show_markers:
            ax.scatter(x, pc, color='red', s=10, marker=marker, label="Data Points")
        
        # Set title
        if title is None:
            title = f"Principal Component {mode_number}"
        ax.set_title(title, fontsize=12)
        
        # Add labels
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("PC Amplitude", fontsize=12)
        
        # Add grid
        if grid:
            ax.grid(True, linestyle="--", alpha=0.7)
        
        # Add legend
        ax.legend(fontsize=10)
        
        # If dates are datetime, adjust x-axis
        if isinstance(x, pd.DatetimeIndex):
            fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    except Exception as e:
        raise VisualizationError(f"Error plotting PC: {str(e)}")


def plot_multiple_pcs(pcs, dates=None, mode_numbers=None,
                   titles=None, colors=None, figsize=(15, 10),
                   grid=True, show_markers=True, date_format=None,
                   shared_x=True, save_path=None):
    """
    Plot multiple Principal Components (PCs) as time series.
    
    Parameters
    ----------
    pcs : ndarray
        2D array containing multiple PCs (columns=modes, rows=time steps).
    dates : list or ndarray, optional
        Dates or time values corresponding to PC values.
        If None, uses index values.
    mode_numbers : list, optional
        List of mode numbers. If None, uses [1, 2, ..., n].
    titles : list, optional
        List of titles for each subplot. If None, default titles are generated.
    colors : list, optional
        List of colors for each PC. If None, uses default colors.
    figsize : tuple, optional
        Figure size. Default is (15, 10).
    grid : bool, optional
        Whether to show grid. Default is True.
    show_markers : bool, optional
        Whether to show markers. Default is True.
    date_format : str, optional
        Format string for parsing dates (e.g., '%Y%m' for '202301'). 
        Default is None (auto-detect).
    shared_x : bool, optional
        Whether to share x-axis. Default is True.
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved.
        
    Returns
    -------
    tuple
        (fig, axes) tuple with the figure and axes objects.
    """
    try:
        # Determine the number of PCs
        if len(pcs.shape) == 1:
            n_pcs = 1
            pcs = pcs.reshape(-1, 1)
        else:
            if pcs.shape[1] > pcs.shape[0]:
                # Transpose if rows are modes and columns are time steps
                pcs = pcs.T
            n_pcs = pcs.shape[1]
        
        # Create figure and axes
        fig, axes = plt.subplots(n_pcs, 1, figsize=figsize, sharex=shared_x)
        
        # Convert axes to array if only one PC
        if n_pcs == 1:
            axes = [axes]
        
        # Create default mode numbers if not provided
        if mode_numbers is None:
            mode_numbers = list(range(1, n_pcs + 1))
        
        # Create default titles if not provided
        if titles is None:
            titles = [f"Principal Component {mode}" for mode in mode_numbers]
        
        # Create default colors if not provided
        if colors is None:
            colors = plt.cm.tab10.colors
            if n_pcs > len(colors):
                # Cycle through colors if more PCs than colors
                colors = [colors[i % len(colors)] for i in range(n_pcs)]
        elif len(colors) < n_pcs:
            # Extend colors if not enough provided
            colors.extend([plt.cm.tab10.colors[i % len(plt.cm.tab10.colors)] 
                          for i in range(len(colors), n_pcs)])
        
        # Create x values
        if dates is None:
            x = np.arange(pcs.shape[0])
        else:
            x = dates
            
            # Convert string dates to datetime if needed
            if isinstance(x[0], str):
                try:
                    if date_format:
                        # Use specified format
                        x = pd.to_datetime(x, format=date_format)
                    else:
                        # Try auto-detection
                        x = pd.to_datetime(x)
                except:
                    # If conversion fails, use indices
                    x = np.arange(pcs.shape[0])
        
        # Plot each PC
        for i in range(n_pcs):
            pc = pcs[:, i]
            ax = axes[i]
            
            # Plot PC
            ax.plot(x, pc, color=colors[i], linestyle='-', linewidth=1, label=f"PC {mode_numbers[i]}")
            
            # Add markers if requested
            if show_markers:
                ax.scatter(x, pc, color='red', s=10, marker='.', label="Data Points")
            
            # Set title
            ax.set_title(titles[i], fontsize=12)
            
            # Add labels
            if i == n_pcs - 1:
                ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("PC Amplitude", fontsize=12)
            
            # Add grid
            if grid:
                ax.grid(True, linestyle="--", alpha=0.7)
            
            # Add legend
            ax.legend(fontsize=10)
        
        # If dates are datetime, adjust x-axis
        if isinstance(x, pd.DatetimeIndex):
            fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, axes
    
    except Exception as e:
        raise VisualizationError(f"Error plotting multiple PCs: {str(e)}")