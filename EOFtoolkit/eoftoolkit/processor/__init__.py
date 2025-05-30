"""Processor module for EOFtoolkit."""

from eoftoolkit.processor.dimensions import standardize_dimensions
from eoftoolkit.processor.masking import create_binary_mask, create_super_mask
from eoftoolkit.processor.identification import create_id_matrix
from eoftoolkit.processor.flattener import flatten_matrices
from eoftoolkit.processor.reshaper import reshape_to_spatial_grid
from eoftoolkit.processor.stacker import create_super_matrix

__all__ = [
    'standardize_dimensions',
    'create_binary_mask',
    'create_super_mask',
    'create_id_matrix',
    'flatten_matrices',
    'reshape_to_spatial_grid',
    'create_super_matrix'
]