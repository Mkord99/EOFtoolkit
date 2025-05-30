"""Visualization module for EOFtoolkit."""

from eoftoolkit.visualization.eof_plots import plot_eof
from eoftoolkit.visualization.pc_plots import plot_pc
from eoftoolkit.visualization.reconstruction_plots import plot_reconstruction
from eoftoolkit.visualization.error_plots import plot_reconstruction_error
from eoftoolkit.visualization.spatial import plot_reconstruction_comparison

__all__ = [
    'plot_eof',
    'plot_pc',
    'plot_reconstruction',
    'plot_reconstruction_error',
    'plot_reconstruction_comparison'
]