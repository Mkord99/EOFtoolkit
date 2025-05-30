"""Module for creating super matrices by stacking flattened matrices."""

import numpy as np
from eoftoolkit.core.exceptions import DimensionError


def create_super_matrix(flattened_dict, keys=None):
    """
    Create a super matrix by stacking flattened matrices.
    
    Parameters
    ----------
    flattened_dict : dict
        Dictionary with keys as matrix identifiers and values as flattened arrays.
    keys : list, optional
        List of keys to include in the super matrix. If None, uses all keys sorted
        in ascending order (useful for chronological ordering).
        
    Returns
    -------
    ndarray
        Super matrix with rows corresponding to matrices and columns to spatial locations.
    list
        List of keys in the order they were stacked.
    """
    if not flattened_dict:
        raise DimensionError("No flattened matrices provided")
    
    # Determine which keys to use
    if keys is None:
        keys = sorted(flattened_dict.keys())
    
    # Check if all specified keys exist
    for key in keys:
        if key not in flattened_dict:
            raise KeyError(f"Key '{key}' not found in flattened_dict")
    
    # Check if all matrices have the same number of columns
    first_shape = flattened_dict[keys[0]].shape
    for key in keys:
        if flattened_dict[key].shape != first_shape:
            raise DimensionError(f"Matrix '{key}' has shape {flattened_dict[key].shape}, "
                                f"but expected {first_shape}")
    
    # Stack matrices vertically
    matrices_to_stack = [flattened_dict[key] for key in keys]
    
    # Ensure matrices are 2D
    for i, matrix in enumerate(matrices_to_stack):
        if len(matrix.shape) == 1:
            matrices_to_stack[i] = matrix.reshape(1, -1)
    
    super_matrix = np.vstack(matrices_to_stack)
    
    return super_matrix, keys