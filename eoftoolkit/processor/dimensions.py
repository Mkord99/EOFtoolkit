"""Module for standardizing matrix dimensions."""

import numpy as np
from eoftoolkit.core.exceptions import DimensionError


def standardize_dimensions(matrices_dict, target_dims=None):
    """
    Standardize dimensions of all matrices.
    
    Parameters
    ----------
    matrices_dict : dict
        Dictionary with keys as matrix identifiers and values as 2D arrays.
    target_dims : tuple, optional
        Target dimensions as (rows, cols). If None, uses maximum dimensions
        found in the input matrices.
        
    Returns
    -------
    dict
        Dictionary with same keys and standardized matrices as values.
    tuple
        The target dimensions used (rows, cols).
    """
    if not matrices_dict:
        raise DimensionError("No matrices provided for standardization")
    
    # Find maximum dimensions if target_dims is not provided
    if target_dims is None:
        max_rows = 0
        max_cols = 0
        
        for matrix in matrices_dict.values():
            if matrix is None or not hasattr(matrix, 'shape'):
                continue
            
            # Handle different dimension cases
            if len(matrix.shape) == 1:
                rows = 1
                cols = matrix.shape[0]
            else:
                rows, cols = matrix.shape[:2]  # Take first two dimensions
                
            max_rows = max(max_rows, rows)
            max_cols = max(max_cols, cols)
            
        target_dims = (max_rows, max_cols)
    
    if target_dims[0] <= 0 or target_dims[1] <= 0:
        raise DimensionError("Invalid target dimensions")
    
    # Standardize each matrix
    standardized_dict = {}
    for key, matrix in matrices_dict.items():
        if matrix is None:
            standardized_dict[key] = None
            continue
            
        if not hasattr(matrix, 'shape'):
            raise DimensionError(f"Matrix '{key}' is not a valid array")
        
        # Handle different dimension cases
        if len(matrix.shape) == 1:
            matrix = matrix.reshape(1, -1)
            
        rows, cols = matrix.shape[:2]
        
        # Create a new matrix filled with NaN values
        standardized = np.full(target_dims, np.nan, dtype=np.float64)
        
        # Copy existing data
        standardized[:rows, :cols] = matrix
        
        # Store standardized matrix
        standardized_dict[key] = standardized
    
    return standardized_dict, target_dims