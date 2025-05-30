"""Module for writing output files."""

import os
import numpy as np


def save_results(result_dict, output_dir, prefix='eof_analysis'):
    """
    Save analysis results to files.
    
    Parameters
    ----------
    result_dict : dict
        Dictionary containing analysis results.
    output_dir : str
        Directory to save the files.
    prefix : str, optional
        Prefix for the output filenames.
        
    Returns
    -------
    dict
        Dictionary with paths to saved files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    saved_files = {}
    
    # Save EOFs
    if 'eofs' in result_dict:
        eofs_path = os.path.join(output_dir, f"{prefix}_eofs.npy")
        np.save(eofs_path, result_dict['eofs'])
        saved_files['eofs'] = eofs_path
    
    # Save PCs
    if 'pcs' in result_dict:
        pcs_path = os.path.join(output_dir, f"{prefix}_pcs.npy")
        np.save(pcs_path, result_dict['pcs'])
        saved_files['pcs'] = pcs_path
    
    # Save reconstruction
    if 'reconstruction' in result_dict:
        recon_path = os.path.join(output_dir, f"{prefix}_reconstruction.npy")
        np.save(recon_path, result_dict['reconstruction'])
        saved_files['reconstruction'] = recon_path
    
    # Save ID matrix
    if 'id_matrix' in result_dict:
        id_path = os.path.join(output_dir, f"{prefix}_id_matrix.npy")
        np.save(id_path, result_dict['id_matrix'])
        saved_files['id_matrix'] = id_path
    
    # Save super mask
    if 'super_mask' in result_dict:
        mask_path = os.path.join(output_dir, f"{prefix}_super_mask.npy")
        np.save(mask_path, result_dict['super_mask'])
        saved_files['super_mask'] = mask_path
    
    # Save error metrics
    if 'error_metrics' in result_dict:
        metrics_path = os.path.join(output_dir, f"{prefix}_error_metrics.npy")
        np.save(metrics_path, result_dict['error_metrics'])
        saved_files['error_metrics'] = metrics_path
    
    return saved_files