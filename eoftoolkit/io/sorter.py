"""Module for sorting files by date."""

import os
import re
from datetime import datetime

def sort_files_by_date(directory_path, file_extension='.nc', date_pattern=None, date_format=None):
    """
    Sort files in a directory by date embedded in filenames.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory containing files.
    file_extension : str, optional
        Extension of files to process. Default is '.nc'.
    date_pattern : str, optional
        Regular expression pattern to extract date from filename.
        If None, it will try to use the filename as a date.
    date_format : str, optional
        Format string for parsing the date if a pattern is provided.
        For example, '%Y%m' for dates like '199301'.
        
    Returns
    -------
    list
        List of sorted file paths.
    """
    # Find all files with the specified extension
    files = [f for f in os.listdir(directory_path) if f.endswith(file_extension)]
    
    if not files:
        return []
    
    if date_pattern:
        # Extract dates using the provided pattern
        date_map = {}
        for filename in files:
            match = re.search(date_pattern, filename)
            if match:
                date_str = match.group(1)
                try:
                    if date_format:
                        date_obj = datetime.strptime(date_str, date_format)
                    else:
                        # Try to convert directly to integer if no format
                        date_obj = int(date_str)
                    date_map[filename] = date_obj
                except (ValueError, TypeError):
                    # Skip files that don't match the expected format
                    continue
        
        # Sort files by their extracted dates
        sorted_files = sorted(date_map.keys(), key=lambda x: date_map[x])
    else:
        # If no pattern is provided, try to interpret the filename (without extension) as the date
        date_map = {}
        for filename in files:
            base_name = os.path.splitext(filename)[0]
            try:
                # Try to convert to integer (e.g., '199301')
                date_map[filename] = int(base_name)
            except ValueError:
                # If that fails, keep the file but don't sort it by date
                date_map[filename] = base_name
        
        # Sort files by their interpreted dates
        sorted_files = sorted(date_map.keys(), key=lambda x: date_map[x])
    
    # Return full paths
    return [os.path.join(directory_path, f) for f in sorted_files]