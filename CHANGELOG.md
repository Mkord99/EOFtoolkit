# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-30

### Added
- Initial release of EOFtoolkit
- Core EOF analysis functionality using SVD
- EOFProcessor class for complete workflow management
- Support for NetCDF file reading and processing
- Comprehensive data preprocessing pipeline including:
  - Dimension standardization
  - Binary and super mask creation
  - Matrix flattening and centering
  - Super matrix construction
- SVD analysis with configurable number of modes
- Data reconstruction with optimal mode selection
- Error metrics calculation (RMSE, MAE, R²)
- Visualization capabilities for:
  - EOF spatial patterns
  - Principal component time series
  - Reconstructed data
  - Reconstruction errors
  - Comparison plots
- Geographic projection support (Mercator, Lambert Conformal Conic, Stereographic)
- Comprehensive test suite with unit, integration, and validation tests
- Support for Python 3.8+

### Features
- Fast and memory-efficient EOF decomposition
- Automatic optimal reconstruction mode selection
- Multiple map projections for visualization
- Extensive error handling and custom exceptions
- Date-based file sorting and filtering
- Configurable visualization parameters
- Scientific accuracy validation against known patterns

### Documentation
- Complete API documentation
- Usage examples and tutorials
- Comprehensive README with quickstart guide
- Scientific methodology explanations
