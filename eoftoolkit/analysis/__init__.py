"""Analysis module for EOFtoolkit."""

from eoftoolkit.analysis.svd import perform_svd
from eoftoolkit.analysis.reconstruction import reconstruct_from_modes
from eoftoolkit.analysis.validation import calculate_error_metrics

# Expose simplified API functions
svd = perform_svd
reconstruct = reconstruct_from_modes

__all__ = [
    'perform_svd',
    'reconstruct_from_modes',
    'calculate_error_metrics',
    'svd',
    'reconstruct'
]