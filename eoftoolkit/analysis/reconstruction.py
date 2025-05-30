"""Module for reconstructing data from SVD results."""

import numpy as np
from eoftoolkit.core.exceptions import ReconstructionError
from eoftoolkit.analysis.validation import calculate_error_metrics


def reconstruct_from_modes(svd_results, max_modes=None, metric='rmse'):
    """
    Perform incremental reconstruction using SVD modes and find optimal reconstruction.
    
    Parameters
    ----------
    svd_results : dict
        Results from perform_svd function.
    max_modes : int, optional
        Maximum number of modes to use in reconstruction. If None, uses all available modes.
    metric : str, optional
        Metric to use for determining optimal reconstruction ('rmse', 'mae', 'mse').
        Default is 'rmse'.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'reconstructions': Incremental reconstructions
        - 'optimal_reconstruction': Optimal reconstruction
        - 'optimal_mode_count': Number of modes in optimal reconstruction
        - 'error_metrics': Error metrics for each reconstruction
    """
    try:
        # Extract necessary components from SVD results
        eofs = svd_results['eofs']
        pcs = svd_results['pcs']
        
        # Determine number of modes to use
        if max_modes is None:
            max_modes = eofs.shape[0]
        else:
            max_modes = min(max_modes, eofs.shape[0])
        
        # Create incremental reconstructions
        reconstructions = {}
        current_reconstruction = np.zeros_like(pcs[:, 0].reshape(-1, 1) @ eofs[0, :].reshape(1, -1))
        
        for i in range(max_modes):
            eof = eofs[i, :].reshape(1, -1)
            pc = pcs[:, i].reshape(-1, 1)
            
            # Create corresponding surface
            surface = pc @ eof
            
            # Add to current reconstruction
            current_reconstruction = current_reconstruction + surface
            
            # Store this reconstruction
            reconstructions[i+1] = current_reconstruction.copy()
        
        # Calculate error metrics if the original super matrix is available in svd_results
        error_metrics = {}
        if 'super_matrix' in svd_results:
            original = svd_results['super_matrix']
            
            for i, reconstruction in reconstructions.items():
                errors = calculate_error_metrics(original, reconstruction)
                error_metrics[i] = errors
            
            # Find optimal reconstruction
            if metric == 'rmse':
                optimal_mode_count = min(error_metrics, key=lambda x: error_metrics[x]['rmse'])
            elif metric == 'mae':
                optimal_mode_count = min(error_metrics, key=lambda x: error_metrics[x]['mae'])
            elif metric == 'mse':
                optimal_mode_count = min(error_metrics, key=lambda x: error_metrics[x]['mse'])
            else:
                optimal_mode_count = max(error_metrics.keys())
        else:
            # If original not available, use all modes
            optimal_mode_count = max(reconstructions.keys())
        
        optimal_reconstruction = reconstructions[optimal_mode_count]
        
        return {
            'reconstructions': reconstructions,
            'optimal_reconstruction': optimal_reconstruction,
            'optimal_mode_count': optimal_mode_count,
            'error_metrics': error_metrics
        }
    
    except Exception as e:
        raise ReconstructionError(f"Error during reconstruction: {str(e)}")


def add_means_back(reconstructions, mean_dict, keys):
    """
    Add back the means that were subtracted during centering.
    
    Parameters
    ----------
    reconstructions : dict
        Dictionary with reconstruction matrices.
    mean_dict : dict
        Dictionary with mean values.
    keys : list
        List of keys in the order they were stacked.
        
    Returns
    -------
    dict
        Dictionary with reconstructions with means added back.
    """
    reconstructions_with_means = {}
    
    for recon_key, reconstruction in reconstructions.items():
        recon_with_means = reconstruction.copy()
        
        for i, key in enumerate(keys):
            if key in mean_dict:
                recon_with_means[i, :] = recon_with_means[i, :] + mean_dict[key]
        
        reconstructions_with_means[recon_key] = recon_with_means
    
    return reconstructions_with_means