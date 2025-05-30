"""I/O module for EOFtoolkit."""

from eoftoolkit.io.reader import read_netcdf
from eoftoolkit.io.sorter import sort_files_by_date

__all__ = ['read_netcdf', 'sort_files_by_date']