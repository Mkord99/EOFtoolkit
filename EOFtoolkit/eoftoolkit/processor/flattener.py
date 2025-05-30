"""Module for flattening matrices based on ID matrix."""

import numpy as np


def flatten_matrices(matrices_dict, id_matrix, super_mask=None):
    """
    Flatten 2D spatial matrices to 1D vectors, keeping only valid cells.
    
    Parameters
    ----------
    matrices_dict : dict
        Dictionary with keys as matrix identifiers and values as 2D arrays.
    id_matrix : ndarray
        ID matrix with IDs for valid cells.
    super_mask : ndarray, optional
        Super mask with 1 for valid cells. If None, inferred from id_matrix.
        
    Returns
    -------
    dict
        Dictionary with same keys and flattened 1D arrays as values.
    ndarray
        Flattened ID matrix.
    """
    # If super_mask is not provided, infer it from id_matrix
    if super_mask is None:
        super_mask = np.where(id_matrix != '', 1, 0)
    
    # Flatten ID matrix
    fl_id_matrix = id_matrix[super_mask == 1].reshape(1, -1)
    
    # Flatten each matrix
    flattened_dict = {}
    for key, matrix in matrices_dict.items():
        if isinstance(matrix, np.ma.MaskedArray):
            # For masked arrays, extract only unmasked values in valid cells
            # First create a view to avoid modifying the original
            matrix_view = matrix.copy()
            # Apply super_mask to the data (not the mask)
            values = matrix_view.data[super_mask == 1]
            # Keep only values that are not masked
            values = values[~matrix_view.mask.flat[super_mask.flat == 1]]
            flattened = values.reshape(1, -1)
        else:
            # For regular arrays, extract values in valid cells
            flattened = matrix[super_mask == 1].reshape(1, -1)
        
        flattened_dict[key] = flattened
    
    return flattened_dict, fl_id_matrix


def center_matrices(flattened_dict, axis=1, return_means=False):
    """
    Center flattened matrices by subtracting the mean.
    
    Parameters
    ----------
    flattened_dict : dict
        Dictionary with keys as matrix identifiers and values as flattened arrays.
    axis : int, optional
        Axis along which to compute the mean (0=columns, 1=rows). Default is 1.
    return_means : bool, optional
        Whether to return the means that were subtracted. Default is False.
        
    Returns
    -------
    dict
        Dictionary with same keys and centered arrays as values.
    dict, optional
        Dictionary with the mean values if return_means is True.
    """
    centered_dict = {}
    mean_dict = {}
    
    for key, flattened in flattened_dict.items():
        mean_value = np.mean(flattened, axis=axis, keepdims=True)
        centered = flattened - mean_value
        
        centered_dict[key] = centered
        mean_dict[key] = mean_value
    
    if return_means:
        return centered_dict, mean_dict
    else:
        return centered_dict