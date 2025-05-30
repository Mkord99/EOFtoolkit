# EOFtoolkit

A comprehensive Python toolkit for **Empirical Orthogonal Function (EOF)** analysis, designed for climate scientists, meteorologists, oceanographers, and researchers working with spatiotemporal data analysis.

## 🌟 Features

- **Fast EOF Decomposition**: Efficient computation of EOFs using SVD and eigenvalue decomposition
- **Multiple EOF Methods**: Standard EOF, Rotated EOF (REOF), Complex EOF (CEOF), and Extended EOF (EEOF)
- **Statistical Analysis**: Significance testing, North's rule of thumb, and Monte Carlo methods
- **Reconstruction Tools**: Time series reconstruction and pattern filtering
- **Visualization**: Built-in plotting functions for EOF patterns and principal components
- **Data Handling**: Support for NetCDF, CSV, and common geospatial data formats
- **Performance Optimized**: Leverages NumPy, SciPy, and optional Dask for large datasets

## 📦 Installation

### Via pip (PyPI)

```bash
pip install eoftoolkit
```

### Via conda (conda-forge)

```bash
conda install -c conda-forge eoftoolkit
```

### Development Installation

```bash
git clone https://github.com/yourusername/EOFtoolkit.git
cd EOFtoolkit
pip install -e .
```

## 🚀 Quick Start

```python
import numpy as np
from eoftoolkit import EOF, load_sample_data

# Load sample climate data
data = load_sample_data('sst_anomalies')  # Sea surface temperature anomalies

# Initialize EOF analysis
eof_analysis = EOF(data)

# Compute EOFs
eof_analysis.solve()

# Get the first 3 EOF patterns and principal components
patterns = eof_analysis.patterns(neofs=3)
pcs = eof_analysis.pcs(neofs=3)
explained_variance = eof_analysis.explained_variance_ratio(neofs=3)

print(f"Explained variance: {explained_variance}")

# Plot results
eof_analysis.plot_patterns(neofs=3)
eof_analysis.plot_pcs(neofs=3)
```

## 📚 Core Functionality

### EOF Analysis

```python
from eoftoolkit import EOF

# Standard EOF analysis
eof = EOF(data, weights=None, center=True, ddof=1)
eof.solve()

# Access results
patterns = eof.patterns()           # EOF spatial patterns
pcs = eof.pcs()                    # Principal components (time series)
eigenvalues = eof.eigenvalues()    # Eigenvalues
explained_var = eof.explained_variance_ratio()
```

### Rotated EOF (REOF)

```python
from eoftoolkit import REOF

# Rotated EOF analysis
reof = REOF(data, neofs=10)
reof.solve(method='varimax')

# Get rotated patterns
rotated_patterns = reof.patterns()
rotated_pcs = reof.pcs()
```

### Complex EOF (CEOF)

```python
from eoftoolkit import CEOF

# Complex EOF for oscillatory patterns
ceof = CEOF(data, tau=1)  # tau is the lag
ceof.solve()

# Access complex patterns
amplitude = ceof.amplitude()
phase = ceof.phase()
```

### Statistical Significance

```python
# North's rule of thumb for EOF significance
significant_modes = eof.north_significance_test()

# Monte Carlo significance testing
p_values = eof.monte_carlo_test(n_trials=1000)
```

## 🌍 Use Cases

### Climate Science
- **ENSO Analysis**: Identify El Niño and La Niña patterns in sea surface temperatures
- **NAO/AO Studies**: Extract North Atlantic or Arctic Oscillation patterns
- **Precipitation Patterns**: Analyze dominant modes of precipitation variability

### Meteorology
- **Storm Track Analysis**: Identify preferred storm paths and intensity patterns
- **Temperature Trends**: Decompose temperature fields into leading modes of variability
- **Pressure Systems**: Analyze atmospheric pressure patterns and teleconnections

### Oceanography
- **Current Systems**: Study dominant patterns in ocean circulation
- **Heat Content**: Analyze ocean heat content variability
- **Upwelling Patterns**: Identify coastal upwelling modes

## 🔧 Dependencies

### Required
- Python >= 3.12
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- pandas >= 1.5.0

### Optional
- matplotlib >= 3.6.0 (for plotting)
- cartopy >= 0.21.0 (for geospatial plotting)
- xarray >= 2023.1.0 (for NetCDF support)
- dask >= 2023.1.0 (for large dataset handling)
- netCDF4 >= 1.6.0 (for NetCDF I/O)

## 🧪 Examples

### Example 1: Sea Surface Temperature Analysis

```python
import xarray as xr
from eoftoolkit import EOF

# Load SST data
sst = xr.open_dataset('sst_data.nc')['sst']

# Prepare data (remove seasonal cycle, etc.)
sst_anomalies = sst.groupby('time.month') - sst.groupby('time.month').mean()

# EOF analysis
eof = EOF(sst_anomalies.values)
eof.solve()

# Plot the first EOF pattern
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.contourf(eof.patterns()[0])
plt.title('First EOF Pattern - SST Anomalies')
plt.colorbar(label='EOF Loading')
plt.show()
```

### Example 2: Significance Testing

```python
# Determine significant EOFs using North's rule
n_significant = eof.north_significance_test()
print(f"Number of significant EOFs: {n_significant}")

# Detailed significance testing
eigenvalues = eof.eigenvalues()
errors = eof.north_errors()

for i in range(min(10, len(eigenvalues))):
    if eigenvalues[i] > errors[i]:
        print(f"EOF {i+1}: Significant (λ={eigenvalues[i]:.3f}, error={errors[i]:.3f})")
    else:
        print(f"EOF {i+1}: Not significant")
```

### Development Setup

```bash
git clone https://github.com/yourusername/EOFtoolkit.git
cd EOFtoolkit
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Mkord99/EOFtoolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Mkord99/EOFtoolkit/discussions)
- **Email**: mohammadkord99@gmail.com

## 📋 Changelog

### Version 0.0.1 (2025-01-17)
- Initial release
- Basic EOF functionality
- Standard, Rotated, and Complex EOF methods
- Significance testing capabilities
- Basic visualization tools

## 🙏 Acknowledgments

- Inspired by the [eofs](https://github.com/ajdawson/eofs) package by Andrew Dawson
- EOF methodology based on Preisendorfer (1988) and North et al. (1982)
- Thanks to the scientific Python community for excellent foundational packages

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made for my Geoinformatics Project at Politecnico di Milano**
