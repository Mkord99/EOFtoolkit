"""Custom exceptions for EOFtoolkit."""

class EOFToolkitError(Exception):
    """Base exception for all EOFtoolkit errors."""
    pass


class FileReadError(EOFToolkitError):
    """Exception raised when a file cannot be read."""
    pass


class DimensionError(EOFToolkitError):
    """Exception raised when there is an issue with matrix dimensions."""
    pass


class SVDError(EOFToolkitError):
    """Exception raised during SVD computation."""
    pass


class ReconstructionError(EOFToolkitError):
    """Exception raised during data reconstruction."""
    pass


class VisualizationError(EOFToolkitError):
    """Exception raised during data visualization."""
    pass