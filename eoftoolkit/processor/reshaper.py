"""Module for reshaping flattened data back to spatial grids."""

import numpy as np


def reshape_to_spatial_grid(flattened_data, id_matrix, target_dims=None, flip_y=True):
    """
    Reshape flattened data back to 2D spatial grid using the ID matrix.
    
    Parameters
    ----------
    flattened_data : ndarray
        1D array of flattened data.
    id_matrix : ndarray
        ID matrix used for reshaping.
    target_dims : tuple, optional
        Target dimensions as (rows, cols). If None, uses dimensions of id_matrix.
    flip_y : bool, optional
        Whether to flip the data vertically (along y-axis) after reshaping.
        This is often needed for geographic data to match the proper orientation.
        Default is True.
        
    Returns
    -------
    ndarray
        Reshaped 2D spatial grid.
    """
    if target_dims is None:
        target_dims = id_matrix.shape
    
    # Create empty grid filled with NaN - ENSURE float64 dtype
    reshaped_grid = np.full(target_dims, np.nan, dtype=np.float64)
    
    # Get flattened ID matrix to use as a reference
    fl_id_values = id_matrix[id_matrix != ''].flatten()
    
    # Make sure flattened_data is 1D
    if len(flattened_data.shape) > 1:
        flattened_data = flattened_data.flatten()
    
    # Check that the lengths match
    if len(fl_id_values) != len(flattened_data):
        raise ValueError(f"Length mismatch: flattened_data has {len(flattened_data)} elements, "
                         f"but ID matrix has {len(fl_id_values)} valid cells")
    
    # Map the values back to the grid based on the ID
    for idx, cell_id in enumerate(fl_id_values):
        try:
            # Ensure cell_id is a string
            cell_id = str(cell_id)
            
            # Extract row and column indices from the ID
            if len(cell_id) >= 4:
                # Parse the ID (format should be RRCC)
                row = int(cell_id[:2]) - 1  # Convert to 0-based index
                col = int(cell_id[2:4]) - 1  # Convert to 0-based index
            else:
                # Handle unexpected ID format
                continue
                
        except (ValueError, IndexError) as e:
            # If parsing fails, skip this ID
            continue
        
        # Assign the value to the grid (with bounds checking)
        if 0 <= row < target_dims[0] and 0 <= col < target_dims[1]:
            reshaped_grid[row, col] = flattened_data[idx]
    
    #print(f"Before flip: reshaped_grid[0,0] = {reshaped_grid[0,0]}")
    #print(f"Before flip: reshaped_grid shape = {reshaped_grid.shape}")
    
    # Flip the grid vertically if needed to match the correct orientation
    if flip_y:
        reshaped_grid = np.flipud(reshaped_grid)
    
    return reshaped_grid


def reshape_all_to_spatial_grid(flattened_dict, id_matrix, target_dims=None, flip_y=True):
    """
    Reshape all flattened data in a dictionary back to 2D spatial grids.
    
    Parameters
    ----------
    flattened_dict : dict
        Dictionary with keys as identifiers and values as flattened arrays.
    id_matrix : ndarray
        ID matrix used for reshaping.
    target_dims : tuple, optional
        Target dimensions as (rows, cols). If None, uses dimensions of id_matrix.
    flip_y : bool, optional
        Whether to flip the data vertically after reshaping. Default is True.
        
    Returns
    -------
    dict
        Dictionary with same keys and reshaped 2D spatial grids as values.
    """
    reshaped_dict = {}
    
    for key, flattened_data in flattened_dict.items():
        reshaped = reshape_to_spatial_grid(flattened_data, id_matrix, target_dims, flip_y)
        reshaped_dict[key] = reshaped
    
    return reshaped_dict