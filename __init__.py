"""
EOFtoolkit - A comprehensive library for EOF (Empirical Orthogonal Function) analysis of NetCDF data.

This library processes geographic/spatial-temporal data stored in NetCDF files,
performs SVD analysis to extract Empirical Orthogonal Functions (EOFs) and Principal Components (PCs),
creates optimal data reconstructions, and visualizes results on maps.
"""

from eoftoolkit.version import __version__
from eoftoolkit.core.processor import EOFProcessor
from eoftoolkit.analysis import svd, reconstruct
from eoftoolkit.visualization import (
    plot_eof,
    plot_pc,
    plot_reconstruction,
    plot_reconstruction_error,
    plot_reconstruction_comparison
)

__all__ = [
    '__version__',
    'EOFProcessor',
    'svd',
    'reconstruct',
    'plot_eof',
    'plot_pc',
    'plot_reconstruction',
    'plot_reconstruction_error',
    'plot_reconstruction_comparison'
]