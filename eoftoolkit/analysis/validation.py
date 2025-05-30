"""Module for validating reconstructions."""

import numpy as np


def calculate_error_metrics(original, reconstruction):
    """
    Calculate error metrics between original and reconstructed data.
    
    Parameters
    ----------
    original : ndarray
        Original data matrix.
    reconstruction : ndarray
        Reconstructed data matrix.
        
    Returns
    -------
    dict
        Dictionary containing various error metrics:
        - 'mse': Mean squared error
        - 'rmse': Root mean squared error
        - 'mae': Mean absolute error
        - 'max_error': Maximum absolute error
        - 'explained_variance': Explained variance score
        - 'r2': R-squared coefficient of determination
    """
    # Calculate errors
    diff = original - reconstruction
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))
    max_error = np.max(np.abs(diff))
    
    # Calculate variance-based metrics
    var_original = np.var(original)
    var_diff = np.var(diff)
    explained_variance = 1 - (var_diff / var_original)
    
    # Calculate R-squared (coefficient of determination)
    ss_total = np.sum((original - np.mean(original)) ** 2)
    ss_residual = np.sum(diff ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'explained_variance': explained_variance,
        'r2': r2
    }


def calculate_temporal_error_metrics(original, reconstruction):
    """
    Calculate error metrics for each time step.
    
    Parameters
    ----------
    original : ndarray
        Original data matrix.
    reconstruction : ndarray
        Reconstructed data matrix.
        
    Returns
    -------
    dict
        Dictionary containing error metrics for each time step:
        - 'rmse': Root mean squared error
        - 'mae': Mean absolute error
    """
    # Calculate errors for each time step
    time_steps = original.shape[0]
    rmse_values = np.zeros(time_steps)
    mae_values = np.zeros(time_steps)
    
    for t in range(time_steps):
        diff = original[t, :] - reconstruction[t, :]
        rmse_values[t] = np.sqrt(np.mean(diff ** 2))
        mae_values[t] = np.mean(np.abs(diff))
    
    return {
        'rmse': rmse_values,
        'mae': mae_values
    }


def calculate_spatial_error_metrics(original, reconstruction):
    """
    Calculate error metrics for each spatial location.
    
    Parameters
    ----------
    original : ndarray
        Original data matrix.
    reconstruction : ndarray
        Reconstructed data matrix.
        
    Returns
    -------
    dict
        Dictionary containing error metrics for each spatial location:
        - 'rmse': Root mean squared error
        - 'mae': Mean absolute error
    """
    # Calculate errors for each spatial location
    locations = original.shape[1]
    rmse_values = np.zeros(locations)
    mae_values = np.zeros(locations)
    
    for loc in range(locations):
        diff = original[:, loc] - reconstruction[:, loc]
        rmse_values[loc] = np.sqrt(np.mean(diff ** 2))
        mae_values[loc] = np.mean(np.abs(diff))
    
    return {
        'rmse': rmse_values,
        'mae': mae_values
    }