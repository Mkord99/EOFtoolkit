"""Module for creating ID matrices."""

import numpy as np


def create_id_matrix(super_mask, base=1, format_string='{:02d}{:02d}'):
    """
    Create an ID matrix for cells that have data across all timestamps.
    
    Parameters
    ----------
    super_mask : ndarray
        Super mask with 1 for valid cells and 0 for invalid cells.
    base : int, optional
        Base index for row and column numbering (0 or 1). Default is 1.
    format_string : str, optional
        Format string for creating cell IDs. Default is '{:02d}{:02d}'.
        
    Returns
    -------
    ndarray
        ID matrix with unique IDs for valid cells and empty strings for invalid cells.
    """
    rows, cols = super_mask.shape
    id_matrix = np.full(super_mask.shape, '', dtype=object)
    
    for i in range(rows):
        for j in range(cols):
            if super_mask[i, j] == 1:
                row_id = i + base
                col_id = j + base
                id_matrix[i, j] = format_string.format(row_id, col_id)
    
    return id_matrix


def get_id_coordinates(id_matrix, longitudes, latitudes):
    """
    Get coordinates for each ID in the ID matrix.
    
    Parameters
    ----------
    id_matrix : ndarray
        ID matrix with IDs for valid cells.
    longitudes : ndarray
        2D grid of longitude values.
    latitudes : ndarray
        2D grid of latitude values.
        
    Returns
    -------
    list
        List of dictionaries with 'id', 'longitude', and 'latitude' keys.
    """
    id_coordinates = []
    
    for i in range(id_matrix.shape[0]):
        for j in range(id_matrix.shape[1]):
            if id_matrix[i, j] != '':
                id_coordinates.append({
                    'id': id_matrix[i, j],
                    'longitude': longitudes[i, j],
                    'latitude': latitudes[i, j]
                })
    
    return id_coordinates