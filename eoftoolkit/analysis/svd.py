"""Module for performing SVD analysis on super matrices."""

import numpy as np
from scipy import linalg
from eoftoolkit.core.exceptions import SVDError


def perform_svd(super_matrix, num_modes=None, compute_surfaces=True):
    """
    Perform SVD analysis on the super matrix and extract EOFs and PCs.
    
    Parameters
    ----------
    super_matrix : ndarray
        Super matrix with rows as time steps and columns as spatial locations.
    num_modes : int, optional
        Number of modes to extract. If None, extracts all modes.
    compute_surfaces : bool, optional
        Whether to compute corresponding surfaces. Default is True.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'eofs': Empirical Orthogonal Functions (spatial patterns)
        - 'pcs': Principal Components (temporal patterns)
        - 'singular_values': Singular values from SVD
        - 'explained_variance': Percentage of variance explained by each mode
        - 'cumulative_variance': Cumulative percentage of variance explained
        - 'corresponding_surfaces': Corresponding surfaces (if compute_surfaces is True)
    """
    
    try:
        # Check for zero matrix
        if np.all(super_matrix == 0):
            raise SVDError("Cannot perform SVD on zero matrix")

        # Perform SVD
        U, s, Vt = linalg.svd(super_matrix, full_matrices=False)
        
        # Determine number of modes to keep
        if num_modes is None:
            num_modes = len(s)
        else:
            num_modes = min(num_modes, len(s))
        
        # Extract EOFs and PCs
        eofs = Vt[:num_modes, :]
        pcs = U[:, :num_modes] * s[:num_modes]
        
        # Calculate variance explained
        variance = s**2 / np.sum(s**2) * 100
        cumulative_variance = np.cumsum(variance)
        
        # Prepare results dictionary
        results = {
            'eofs': eofs,
            'pcs': pcs,
            'singular_values': s[:num_modes],
            'explained_variance': variance[:num_modes],
            'cumulative_variance': cumulative_variance[:num_modes]
        }
        
        # Compute corresponding surfaces if requested
        if compute_surfaces:
            corresponding_surfaces = {}
            
            for i in range(num_modes):
                eof = eofs[i, :].reshape(-1, 1)
                pc = pcs[:, i].reshape(-1, 1)
                surface = np.dot(pc, eof.T)
                corresponding_surfaces[f'mode_{i+1}'] = surface
            
            results['corresponding_surfaces'] = corresponding_surfaces
        
        return results
    
    except Exception as e:
        raise SVDError(f"Error during SVD computation: {str(e)}")


def extract_modes(svd_results, modes_to_extract):
    """
    Extract specific modes from SVD results.
    
    Parameters
    ----------
    svd_results : dict
        Results from perform_svd function.
    modes_to_extract : list
        List of mode indices to extract (1-based).
        
    Returns
    -------
    dict
        Dictionary containing the extracted modes.
    """
    extracted = {}
    
    # Extract EOFs
    eofs = svd_results['eofs']
    extracted_eofs = np.vstack([eofs[i-1, :] for i in modes_to_extract])
    
    # Extract PCs
    pcs = svd_results['pcs']
    extracted_pcs = np.hstack([pcs[:, i-1].reshape(-1, 1) for i in modes_to_extract])
    
    # Extract singular values and variance
    singular_values = svd_results['singular_values']
    extracted_values = np.array([singular_values[i-1] for i in modes_to_extract])
    
    # Extract explained variance
    explained_variance = svd_results['explained_variance']
    extracted_variance = np.array([explained_variance[i-1] for i in modes_to_extract])
    
    # Create results
    extracted['eofs'] = extracted_eofs
    extracted['pcs'] = extracted_pcs
    extracted['singular_values'] = extracted_values
    extracted['explained_variance'] = extracted_variance
    
    # Extract corresponding surfaces if available
    if 'corresponding_surfaces' in svd_results:
        extracted_surfaces = {}
        for i, mode in enumerate(modes_to_extract):
            key = f'mode_{mode}'
            if key in svd_results['corresponding_surfaces']:
                extracted_surfaces[key] = svd_results['corresponding_surfaces'][key]
        
        extracted['corresponding_surfaces'] = extracted_surfaces
    
    return extracted