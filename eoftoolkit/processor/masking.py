"""Module for creating binary masks and super masks."""

import numpy as np


def create_binary_mask(matrix):
    """
    Create a binary mask for a matrix.
    
    Parameters
    ----------
    matrix : ndarray
        Input matrix.
        
    Returns
    -------
    ndarray
        Binary mask with 1 for valid data and 0 for NaN/invalid data.
    """
    # Convert masked arrays if needed
    if isinstance(matrix, np.ma.MaskedArray):
        mask = np.where(matrix.mask, 0, 1)
    else:
        mask = np.where(np.isnan(matrix), 0, 1)
    
    return mask


def create_super_mask(mask_dict, threshold=None):
    """
    Create a super mask identifying cells with valid data across matrices.
    
    Parameters
    ----------
    mask_dict : dict
        Dictionary with keys as mask identifiers and values as binary masks.
    threshold : int, optional
        Minimum number of matrices that must have data for a cell to be included.
        If None, all matrices must have data (logical AND of all masks).
        
    Returns
    -------
    ndarray
        Super mask with 1 for cells that meet the threshold criteria.
    """
    # Stack all masks
    masks = list(mask_dict.values())
    masks_sum = np.sum(masks, axis=0)
    
    # If threshold is not provided, use the total number of masks
    if threshold is None:
        threshold = len(masks)
    
    # Create super mask
    super_mask = np.where(masks_sum >= threshold, 1, 0)
    
    return super_mask