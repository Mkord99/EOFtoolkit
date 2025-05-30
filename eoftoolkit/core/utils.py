"""Utility functions for EOFtoolkit."""

import os
import numpy as np
import datetime


def extract_date_from_filename(filename, pattern=None, format_string=None):
    """
    Extract date from filename.
    
    Parameters
    ----------
    filename : str
        Filename to extract date from.
    pattern : str, optional
        Regular expression pattern with a capture group for the date.
        If None, uses the filename without extension.
    format_string : str, optional
        Format string for datetime.strptime. If None, tries to convert to int.
        
    Returns
    -------
    str or int or datetime
        Extracted date.
    """
    import re
    
    if pattern:
        # Extract using regex
        match = re.search(pattern, filename)
        if match:
            date_str = match.group(1)
        else:
            # If no match, use filename without extension
            date_str = os.path.splitext(filename)[0]
    else:
        # Use filename without extension
        date_str = os.path.splitext(filename)[0]
    
    # Convert to appropriate type
    if format_string:
        try:
            date = datetime.datetime.strptime(date_str, format_string)
            return date
        except ValueError:
            return date_str
    else:
        try:
            date = int(date_str)
            return date
        except ValueError:
            return date_str


def generate_timestamp_labels(num_timestamps, start_date=None, frequency=None):
    """
    Generate timestamp labels.
    
    Parameters
    ----------
    num_timestamps : int
        Number of timestamps.
    start_date : str or datetime, optional
        Start date. If None, uses integers starting from 1.
    frequency : str, optional
        Frequency for date range (e.g., 'M' for month). Default is 'M'.
        
    Returns
    -------
    list
        List of timestamp labels.
    """
    if start_date is None:
        # Use integers
        return list(range(1, num_timestamps + 1))
    else:
        import pandas as pd
        
        # Use date range
        if frequency is None:
            frequency = 'M'  # Default is monthly
        
        dates = pd.date_range(start=start_date, periods=num_timestamps, freq=frequency)
        return dates.tolist()


def validate_directory(directory_path):
    """
    Validate that a directory exists.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory.
        
    Returns
    -------
    bool
        True if the directory exists, False otherwise.
    """
    return os.path.isdir(directory_path)


def find_netcdf_files(directory_path, extension='.nc'):
    """
    Find NetCDF files in a directory.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory.
    extension : str, optional
        File extension to look for. Default is '.nc'.
        
    Returns
    -------
    list
        List of NetCDF file paths.
    """
    if not validate_directory(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    file_paths = []
    for file in os.listdir(directory_path):
        if file.endswith(extension):
            file_paths.append(os.path.join(directory_path, file))
    
    return file_paths


def create_date_range(start_date, end_date, frequency='M'):
    """
    Create a date range.
    
    Parameters
    ----------
    start_date : str or datetime
        Start date.
    end_date : str or datetime
        End date.
    frequency : str, optional
        Frequency for date range. Default is 'M' (month).
        
    Returns
    -------
    list
        List of dates.
    """
    import pandas as pd
    
    dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
    return dates.tolist()


def filter_files_by_date_range(file_paths, start_date, end_date, date_pattern=None, 
                             date_format=None, frequency='M'):
    """
    Filter files by date range.
    
    Parameters
    ----------
    file_paths : list
        List of file paths.
    start_date : str or datetime
        Start date.
    end_date : str or datetime
        End date.
    date_pattern : str, optional
        Regular expression pattern with a capture group for the date.
    date_format : str, optional
        Format string for datetime.strptime.
    frequency : str, optional
        Frequency for date range. Default is 'M' (month).
        
    Returns
    -------
    list
        List of filtered file paths.
    """
    import pandas as pd
    
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        try:
            start_date = pd.to_datetime(start_date)
        except ValueError:
            raise ValueError(f"Invalid start_date: {start_date}")
    
    if isinstance(end_date, str):
        try:
            end_date = pd.to_datetime(end_date)
        except ValueError:
            raise ValueError(f"Invalid end_date: {end_date}")
    
    # Create date range
    if frequency == 'M':
        frequency = 'ME'  # Use 'ME' for month end
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    
    # Filter files
    filtered_paths = []
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        file_date = extract_date_from_filename(filename, date_pattern, date_format)
        
        # Convert to datetime if it's not already
        if not isinstance(file_date, datetime.datetime):
            try:
                if isinstance(file_date, int):
                    # Try to convert integer to string and then to datetime
                    file_date = pd.to_datetime(str(file_date), format=date_format)
                else:
                    file_date = pd.to_datetime(file_date)
            except ValueError:
                # Skip files with invalid dates
                continue
        
        # Check if file date is in range
        if start_date <= file_date <= end_date:
            filtered_paths.append(file_path)
    
    return filtered_paths


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=50):
    """
    Print a progress bar.
    
    Parameters
    ----------
    iteration : int
        Current iteration.
    total : int
        Total iterations.
    prefix : str, optional
        Prefix string.
    suffix : str, optional
        Suffix string.
    decimals : int, optional
        Decimal places for percentage. Default is 1.
    bar_length : int, optional
        Length of the progress bar. Default is 50.
    """
    format_str = "{0:." + str(decimals) + "f}"
    percent = format_str.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    
    # Print new line on complete
    if iteration == total:
        print()