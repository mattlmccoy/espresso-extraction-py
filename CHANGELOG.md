# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-01-30

### Added
- **Command-line interface** with argparse for direct access to features
  - `--collect` flag for data collection
  - `--plot-2d` flag for 2D visualization
  - `--plot-3d` flag for 3D visualization
  - `--data-dir` flag for custom data directory
  - `--help` for usage information
- **Configuration file** (`config.json`) for customizable defaults
  - Default values for temperature, pressure, grind size, and extraction time
  - Grind size to micron mapping
  - Brix to TDS conversion factor
  - Visualization parameters (DPI, trend points)
- **Type hints** throughout the codebase for better IDE support and code clarity
- **Statistical metrics** in visualizations
  - R² (coefficient of determination) for regression quality
  - RMSE (root mean squared error) for prediction accuracy
- **Input validation** functions
  - `validate_positive()` for ensuring positive numeric values
  - `validate_grind_size()` for validating grind size options
- **Enhanced error handling**
  - Try-except blocks for data loading
  - Keyboard interrupt handling for graceful exits
  - Better error messages with context
- **Improved user experience**
  - Real-time feedback showing calculated TDS and extraction yield
  - Better formatted interactive menu
  - Enhanced plot styling with improved colors and labels
  - Optimized 3D viewing angle
- **Project infrastructure**
  - `.gitignore` for Python projects
  - `requirements.txt` with pinned dependency versions
  - `setup.py` for package installation
  - `CHANGELOG.md` for version tracking

### Changed
- **Refactored data collection function**
  - Added input validation loops for critical parameters
  - Improved user prompts with clearer instructions
  - Better feedback after data saving
- **Enhanced 2D visualization**
  - Increased figure size (8x6 → 10x7)
  - Improved scatter plot styling with edge colors
  - Better trend line visualization
  - Statistical annotation repositioned for clarity
  - Higher quality exports with configurable DPI
- **Enhanced 3D visualization**
  - Increased figure size (10x8 → 12x9)
  - Improved scatter plot styling
  - Better trend plane calculation with statistics
  - Optimized viewing angle (elev=20, azim=45)
  - Enhanced legend with statistical information
- **Updated README.md**
  - Comprehensive feature documentation
  - Installation instructions with multiple methods
  - CLI usage examples
  - Configuration guide
  - Updated repository structure
  - Added "What's New in v2.0" section
- **Dependencies updated**
  - Added `scikit-learn>=1.3.0` for statistical metrics
  - Specified minimum versions for all dependencies

### Fixed
- **CSV header order** now matches data collection sequence
- **Removed unused imports**
  - Removed `scipy.interpolate.griddata` (was imported but never used)
  - Removed explicit `Axes3D` import (handled implicitly by matplotlib)
- **NaN value handling** in visualization functions
- **Better error messages** for invalid inputs

### Improved
- **Documentation**
  - Comprehensive docstrings for all functions
  - Parameter descriptions with types
  - Return value documentation
  - Usage examples in README
- **Code quality**
  - Consistent formatting and style
  - Better variable names
  - Modular function design
  - Type safety with type hints

## [1.0.0] - 2025-03-12

### Initial Release
- Basic espresso extraction data collection
- 2D and 3D visualization with linear regression
- CSV data storage
- Interactive menu interface
- Sample data included
