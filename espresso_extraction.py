"""
Espresso Extraction Analysis Tool

A metrology-driven approach to analyzing espresso extraction data.
Collects, analyzes, and visualizes the effects of temperature, pressure,
grind size, and extraction time on espresso extraction yield.
"""

import os
import csv
import glob
import json
import argparse
from datetime import datetime
from typing import Tuple, Optional, Any, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error

# Directory for storing data files
DATA_DIR = "extraction_data"
CONFIG_FILE = "config.json"

# Load configuration
def load_config() -> Dict[str, Any]:
    """Load configuration from JSON file or return defaults."""
    default_config = {
        "defaults": {
            "temperature_celsius": 93,
            "pressure_bar": 9,
            "grind_size": "medium",
            "extraction_time_seconds": 25
        },
        "grind_size_mapping": {
            "fine": 200,
            "medium": 400,
            "coarse": 600
        },
        "brix_to_tds_factor": 0.85,
        "visualization": {
            "figure_dpi": 600,
            "trend_points_2d": 50,
            "trend_points_3d": 15
        }
    }

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config file. Using defaults. Error: {e}")
            return default_config
    return default_config

CONFIG = load_config()


def get_input_with_default(
    prompt: str,
    default: Any,
    cast_func: type = str,
    validator: Optional[callable] = None
) -> Tuple[Any, bool]:
    """
    Prompt the user for input with a default value and optional validation.

    Args:
        prompt: The prompt to display to the user
        default: The default value if user provides no input
        cast_func: Function to cast the input (e.g., int, float, str)
        validator: Optional function to validate the input

    Returns:
        Tuple of (value, flag) where flag is True if user provided a value
    """
    user_input = input(f"{prompt} (default: {default}): ")
    if user_input.strip() == "":
        return default, False
    else:
        try:
            value = cast_func(user_input)
            if validator and not validator(value):
                print(f"Invalid input. Using default {default}.")
                return default, False
            return value, True
        except Exception as e:
            print(f"Invalid input ({e}). Using default {default}.")
            return default, False


def validate_positive(value: float) -> bool:
    """Validate that a value is positive."""
    return value > 0


def validate_grind_size(value: str) -> bool:
    """Validate that grind size is one of the allowed values."""
    return value.lower() in CONFIG["grind_size_mapping"]


def collect_data() -> None:
    """
    Collect espresso extraction trial data and save it as a CSV file.

    Prompts user for extraction parameters, calculates TDS and extraction yield,
    and saves the data to a numbered CSV file in the extraction_data directory.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n=== Espresso Extraction Data Collection ===\n")

    # 1) Dry coffee weight
    while True:
        try:
            dry_coffee_mass = float(input("Enter weight of dry coffee grounds (g): "))
            if dry_coffee_mass > 0:
                break
            print("Error: Weight must be positive.")
        except ValueError:
            print("Error: Please enter a valid number.")

    # 2) Temperature
    temperature, temp_used = get_input_with_default(
        "Enter extraction temperature (°C)",
        CONFIG["defaults"]["temperature_celsius"],
        float,
        validate_positive
    )

    # 3) Pressure
    pressure, pressure_used = get_input_with_default(
        "Enter extraction pressure (bar)",
        CONFIG["defaults"]["pressure_bar"],
        float,
        validate_positive
    )

    # 4) Grind size
    grind_size, grind_used = get_input_with_default(
        "Enter grind size (fine, medium, coarse)",
        CONFIG["defaults"]["grind_size"],
        str,
        validate_grind_size
    )

    # 5) Extraction time
    extraction_time, time_used = get_input_with_default(
        "Enter extraction time (s)",
        CONFIG["defaults"]["extraction_time_seconds"],
        float,
        validate_positive
    )

    # 6) Weight of pulled shot
    while True:
        try:
            beverage_mass = float(input("Enter weight of pulled shot (g): "))
            if beverage_mass > 0:
                break
            print("Error: Weight must be positive.")
        except ValueError:
            print("Error: Please enter a valid number.")

    # 7) Brix
    while True:
        try:
            brix = float(input("Enter Brix reading (as a percentage): "))
            if 0 <= brix <= 100:
                break
            print("Error: Brix must be between 0 and 100.")
        except ValueError:
            print("Error: Please enter a valid number.")

    # Calculate TDS and extraction yield
    # Brix is treated as a percent, so 25 => 25%. Convert to fraction, then apply factor.
    brix_factor = CONFIG["brix_to_tds_factor"]
    tds = (brix * brix_factor) / 100
    extraction_yield = (tds * beverage_mass) / dry_coffee_mass * 100

    # CSV header matches the order of data collection
    header = [
        "Timestamp",
        "Dry Coffee Mass (g)",
        "Temperature (°C)", "Temperature Used",
        "Pressure (bar)", "Pressure Used",
        "Grind Size", "Grind Size Used",
        "Extraction Time (s)", "Extraction Time Used",
        "Beverage Mass (g)",
        "Brix",
        "TDS",
        "Extraction Yield (%)"
    ]

    row = [
        timestamp,
        dry_coffee_mass,
        temperature, temp_used,
        pressure, pressure_used,
        grind_size, grind_used,
        extraction_time, time_used,
        beverage_mass,
        brix,
        tds,
        extraction_yield
    ]

    file_count = len(glob.glob(os.path.join(DATA_DIR, "extraction_*.csv"))) + 1
    filename = os.path.join(DATA_DIR, f"extraction_{file_count}.csv")

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(row)

    print(f"\n✓ Data saved to {filename}")
    print(f"  TDS: {tds:.4f} ({tds*100:.2f}%)")
    print(f"  Extraction Yield: {extraction_yield:.2f}%")


def visualize_data_2d() -> None:
    """
    Load all extraction trial data and create a 2D scatter plot with regression analysis.

    Allows user to select which parameter to plot against extraction yield,
    fits a linear regression line, and displays statistical metrics.
    """
    files = glob.glob(os.path.join(DATA_DIR, "extraction_*.csv"))
    if not files:
        print("No data files found in extraction_data directory.")
        return

    try:
        all_data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    except Exception as e:
        print(f"Error loading data files: {e}")
        return

    # Map grind size to micron values
    grind_size_map = CONFIG["grind_size_mapping"]
    if "Grind Size" in all_data.columns:
        all_data["Grind Size Numeric"] = all_data["Grind Size"].str.lower().map(grind_size_map)

    print("\nSelect metadata for the x-axis (2D plot):")
    print("1: Dry Coffee Mass (g)")
    print("2: Temperature (°C)")
    print("3: Pressure (bar)")
    print("4: Extraction Time (s)")
    print("5: Grind Size")
    choice = input("Enter your choice (1-5): ").strip()

    if choice == "1":
        x_label = "Dry Coffee Mass (g)"
        x_data = all_data[x_label]
    elif choice == "2":
        x_label = "Temperature (°C)"
        x_data = all_data[x_label]
    elif choice == "3":
        x_label = "Pressure (bar)"
        x_data = all_data[x_label]
    elif choice == "4":
        x_label = "Extraction Time (s)"
        x_data = all_data[x_label]
    elif choice == "5":
        x_label = "Grind Size (µm)"
        x_data = all_data["Grind Size Numeric"]
    else:
        print("Invalid choice. Defaulting to Temperature (°C).")
        x_label = "Temperature (°C)"
        x_data = all_data[x_label]

    y_label = "Extraction Yield (%)"
    y_data = all_data[y_label]

    # Remove any NaN values
    valid_mask = ~(x_data.isna() | y_data.isna())
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]

    if len(x_data) < 2:
        print("Insufficient data points for visualization.")
        return

    plt.figure(figsize=(10, 7))
    plt.scatter(x_data, y_data, alpha=0.7, s=80, color='red',
                edgecolors='black', linewidth=0.5, label='Experimental Data')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(f"{x_label}: Impact on Extraction Yield", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Fit a linear regression trend line using least squares
    coeffs = np.polyfit(x_data, y_data, 1)
    m, c = coeffs
    poly = np.poly1d(coeffs)

    # Calculate statistical metrics
    y_pred = poly(x_data)
    r2 = r2_score(y_data, y_pred)
    mse = mean_squared_error(y_data, y_pred)
    rmse = np.sqrt(mse)

    # Generate trend line points
    trend_points = CONFIG["visualization"]["trend_points_2d"]
    x_fit = np.linspace(x_data.min(), x_data.max(), trend_points)
    y_fit = poly(x_fit)

    # Plot trend line
    plt.plot(x_fit, y_fit, color='lightblue', linestyle='-', marker='o',
             markersize=4, alpha=0.6, label='Trend (Least Squares)')

    # Annotate with equation and statistics
    eq_text = (f"Trend: y = {m:.3f}x + {c:.3f}\n"
               f"R² = {r2:.4f}\n"
               f"RMSE = {rmse:.3f}")
    plt.annotate(eq_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                 horizontalalignment='left', verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=1.5, alpha=0.9))
    plt.legend(loc='best')

    save_plot = input("\nDo you want to save the 2D plot? (y/n): ").strip().lower()
    if save_plot == "y":
        figures_dir = os.path.join(DATA_DIR, "figures")
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        x_min = x_data.min()
        x_max = x_data.max()
        sanitized_x_label = x_label.split()[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(
            figures_dir,
            f"extraction_2D_{sanitized_x_label}_{x_min:.0f}-{x_max:.0f}_{timestamp}.png"
        )
        dpi = CONFIG["visualization"]["figure_dpi"]
        plt.savefig(plot_filename, dpi=dpi, bbox_inches='tight')
        print(f"✓ 2D plot saved as {plot_filename}")

    plt.show()


def visualize_data_3d() -> None:
    """
    Load all extraction trial data and create a 3D scatter plot with regression plane.

    Allows user to select two parameters to plot against extraction yield,
    fits a multiple linear regression plane, and displays statistical metrics.
    """
    files = glob.glob(os.path.join(DATA_DIR, "extraction_*.csv"))
    if not files:
        print("No data files found in extraction_data directory.")
        return

    try:
        all_data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    except Exception as e:
        print(f"Error loading data files: {e}")
        return

    # Map grind size to micron values
    grind_size_map = CONFIG["grind_size_mapping"]
    if "Grind Size" in all_data.columns:
        all_data["Grind Size Numeric"] = all_data["Grind Size"].str.lower().map(grind_size_map)

    # Options for x and y axes
    options = {
        "1": ("Dry Coffee Mass (g)", all_data["Dry Coffee Mass (g)"]),
        "2": ("Temperature (°C)", all_data["Temperature (°C)"]),
        "3": ("Pressure (bar)", all_data["Pressure (bar)"]),
        "4": ("Extraction Time (s)", all_data["Extraction Time (s)"]),
        "5": ("Grind Size (µm)", all_data["Grind Size Numeric"])
    }

    print("\nSelect metadata for the x-axis (3D plot):")
    for key, (label, _) in options.items():
        print(f"{key}: {label}")
    x_choice = input("Enter your choice for x-axis (1-5): ").strip()
    if x_choice not in options:
        print("Invalid choice. Defaulting x-axis to Temperature (°C).")
        x_choice = "2"
    x_label, x_data = options[x_choice]

    print("\nSelect metadata for the y-axis (3D plot):")
    for key, (label, _) in options.items():
        print(f"{key}: {label}")
    y_choice = input("Enter your choice for y-axis (1-5): ").strip()
    if y_choice not in options:
        print("Invalid choice. Defaulting y-axis to Grind Size (µm).")
        y_choice = "5"
    y_label, y_data = options[y_choice]

    # z-axis is Extraction Yield (%)
    z_label = "Extraction Yield (%)"
    z_data = all_data[z_label]

    # Remove any NaN values
    valid_mask = ~(x_data.isna() | y_data.isna() | z_data.isna())
    x_array = np.array(x_data[valid_mask])
    y_array = np.array(y_data[valid_mask])
    z_array = np.array(z_data[valid_mask])

    if len(x_array) < 3:
        print("Insufficient data points for 3D visualization (need at least 3).")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot real data in red
    ax.scatter(x_array, y_array, z_array, color='red', s=60,
               marker='o', edgecolors='black', linewidth=0.5,
               label='Experimental Data', alpha=0.8)

    # Fit a plane using multiple linear regression: z = a*x + b*y + c
    A = np.column_stack((x_array, y_array, np.ones(len(x_array))))
    coeffs, residuals, rank, s = np.linalg.lstsq(A, z_array, rcond=None)
    a, b, c = coeffs

    # Calculate R² for the plane fit
    z_pred = a * x_array + b * y_array + c
    r2 = r2_score(z_array, z_pred)
    mse = mean_squared_error(z_array, z_pred)
    rmse = np.sqrt(mse)

    # Set title
    ax.set_title("Charting the Coffee Extraction Landscape",
                 pad=20, fontsize=14, fontweight='bold')

    # Generate trend plane points
    trend_points = CONFIG["visualization"]["trend_points_3d"]
    x_fit = np.linspace(x_array.min(), x_array.max(), trend_points)
    y_fit = np.linspace(y_array.min(), y_array.max(), trend_points)
    x_fit_grid, y_fit_grid = np.meshgrid(x_fit, y_fit)
    z_fit = a * x_fit_grid + b * y_fit_grid + c

    # Overlay trend plane points
    trend_text = (f"Trend Plane: z = {a:.3f}x + {b:.3f}y + {c:.3f}\n"
                  f"R² = {r2:.4f}, RMSE = {rmse:.3f}")
    ax.scatter(x_fit_grid, y_fit_grid, z_fit, color='lightblue',
               s=30, marker='^', alpha=0.5, label=trend_text)

    ax.set_xlabel(x_label, fontsize=11, labelpad=10)
    ax.set_ylabel(y_label, fontsize=11, labelpad=10)
    ax.set_zlabel(z_label, fontsize=11, labelpad=10)
    ax.legend(loc='upper left', fontsize=9)

    # Improve viewing angle
    ax.view_init(elev=20, azim=45)

    save_plot = input("\nDo you want to save the 3D plot? (y/n): ").strip().lower()
    if save_plot == "y":
        figures_dir = os.path.join(DATA_DIR, "figures")
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        x_min, x_max = x_array.min(), x_array.max()
        y_min, y_max = y_array.min(), y_array.max()
        sanitized_x_label = x_label.split()[0]
        sanitized_y_label = y_label.split()[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(
            figures_dir,
            f"extraction_3D_{sanitized_x_label}_{x_min:.0f}-{x_max:.0f}_"
            f"{sanitized_y_label}_{y_min:.0f}-{y_max:.0f}_{timestamp}.png"
        )
        dpi = CONFIG["visualization"]["figure_dpi"]
        plt.savefig(plot_filename, dpi=dpi, bbox_inches='tight')
        print(f"✓ 3D plot saved as {plot_filename}")

    plt.show()


def load_all_data() -> Optional[pd.DataFrame]:
    """
    Load all extraction data from CSV files.

    Returns:
        DataFrame with all data or None if no files found
    """
    files = glob.glob(os.path.join(DATA_DIR, "extraction_*.csv"))
    if not files:
        print("No data files found in extraction_data directory.")
        return None

    try:
        all_data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        # Map grind size to numeric
        grind_size_map = CONFIG["grind_size_mapping"]
        if "Grind Size" in all_data.columns:
            all_data["Grind Size Numeric"] = all_data["Grind Size"].str.lower().map(grind_size_map)
        return all_data
    except Exception as e:
        print(f"Error loading data files: {e}")
        return None


def export_data() -> None:
    """
    Export all extraction data to Excel and JSON formats with summary statistics.
    """
    all_data = load_all_data()
    if all_data is None:
        return

    export_dir = os.path.join(DATA_DIR, "exports")
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export to Excel with multiple sheets
    excel_filename = os.path.join(export_dir, f"extraction_export_{timestamp}.xlsx")
    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # Raw data
            all_data.to_excel(writer, sheet_name='Raw Data', index=False)

            # Summary statistics
            numeric_cols = all_data.select_dtypes(include=[np.number]).columns
            summary = all_data[numeric_cols].describe()
            summary.to_excel(writer, sheet_name='Summary Statistics')

            # Correlation matrix
            corr = all_data[numeric_cols].corr()
            corr.to_excel(writer, sheet_name='Correlations')

        print(f"✓ Excel export saved: {excel_filename}")
    except Exception as e:
        print(f"Error exporting to Excel: {e}")

    # Export to JSON
    json_filename = os.path.join(export_dir, f"extraction_export_{timestamp}.json")
    try:
        export_data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "total_records": len(all_data),
                "columns": list(all_data.columns)
            },
            "data": all_data.to_dict(orient='records'),
            "summary_statistics": all_data.describe().to_dict()
        }
        with open(json_filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"✓ JSON export saved: {json_filename}")
    except Exception as e:
        print(f"Error exporting to JSON: {e}")


def detect_outliers() -> None:
    """
    Detect outliers in extraction yield using IQR and Z-score methods.
    """
    all_data = load_all_data()
    if all_data is None:
        return

    if len(all_data) < 4:
        print("Insufficient data for outlier detection (need at least 4 points).")
        return

    yield_data = all_data["Extraction Yield (%)"].dropna()

    print("\n" + "="*60)
    print("  📊 Outlier Detection Analysis")
    print("="*60)

    # IQR Method
    Q1 = yield_data.quantile(0.25)
    Q3 = yield_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    iqr_outliers = all_data[(yield_data < lower_bound) | (yield_data > upper_bound)]

    print(f"\n1️⃣  IQR Method (Interquartile Range):")
    print(f"   Q1 (25th percentile): {Q1:.2f}%")
    print(f"   Q3 (75th percentile): {Q3:.2f}%")
    print(f"   IQR: {IQR:.2f}%")
    print(f"   Lower bound: {lower_bound:.2f}%")
    print(f"   Upper bound: {upper_bound:.2f}%")
    print(f"   Outliers detected: {len(iqr_outliers)}")

    if len(iqr_outliers) > 0:
        print("\n   Outlier details:")
        for idx, row in iqr_outliers.iterrows():
            print(f"   - Extraction #{idx+1}: {row['Extraction Yield (%)']:.2f}% "
                  f"(Temp: {row['Temperature (°C)']:.1f}°C, "
                  f"Grind: {row['Grind Size']})")

    # Z-Score Method
    z_scores = np.abs(stats.zscore(yield_data))
    z_threshold = 2.5
    z_outliers = all_data[z_scores > z_threshold]

    print(f"\n2️⃣  Z-Score Method (threshold: {z_threshold}):")
    print(f"   Mean: {yield_data.mean():.2f}%")
    print(f"   Std Dev: {yield_data.std():.2f}%")
    print(f"   Outliers detected: {len(z_outliers)}")

    if len(z_outliers) > 0:
        print("\n   Outlier details:")
        for idx, row in z_outliers.iterrows():
            z_score = z_scores.iloc[idx]
            print(f"   - Extraction #{idx+1}: {row['Extraction Yield (%)']:.2f}% "
                  f"(Z-score: {z_score:.2f})")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot with outliers
    axes[0].boxplot(yield_data, vert=True)
    axes[0].set_ylabel('Extraction Yield (%)', fontsize=12)
    axes[0].set_title('Box Plot - IQR Method', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=lower_bound, color='r', linestyle='--', label=f'Lower: {lower_bound:.1f}%')
    axes[0].axhline(y=upper_bound, color='r', linestyle='--', label=f'Upper: {upper_bound:.1f}%')
    axes[0].legend()

    # Scatter plot with Z-scores
    axes[1].scatter(range(len(yield_data)), yield_data, c=z_scores,
                   cmap='coolwarm', s=100, edgecolors='black', linewidth=0.5)
    axes[1].axhline(y=yield_data.mean(), color='green', linestyle='-',
                   label=f'Mean: {yield_data.mean():.1f}%')
    axes[1].axhline(y=yield_data.mean() + z_threshold*yield_data.std(),
                   color='red', linestyle='--', label=f'+{z_threshold}σ')
    axes[1].axhline(y=yield_data.mean() - z_threshold*yield_data.std(),
                   color='red', linestyle='--', label=f'-{z_threshold}σ')
    axes[1].set_xlabel('Extraction Number', fontsize=12)
    axes[1].set_ylabel('Extraction Yield (%)', fontsize=12)
    axes[1].set_title('Z-Score Method', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
    cbar.set_label('|Z-Score|', fontsize=10)

    plt.tight_layout()

    # Save plot
    save_plot = input("\nDo you want to save the outlier detection plot? (y/n): ").strip().lower()
    if save_plot == "y":
        figures_dir = os.path.join(DATA_DIR, "figures")
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(figures_dir, f"outlier_detection_{timestamp}.png")
        plt.savefig(plot_filename, dpi=CONFIG["visualization"]["figure_dpi"], bbox_inches='tight')
        print(f"✓ Outlier plot saved as {plot_filename}")

    plt.show()


def calculate_uncertainty() -> None:
    """
    Calculate uncertainty propagation for TDS and extraction yield measurements.
    """
    all_data = load_all_data()
    if all_data is None:
        return

    print("\n" + "="*60)
    print("  🔬 Uncertainty Propagation Analysis")
    print("="*60)

    # Ask for measurement uncertainties
    print("\nEnter measurement uncertainties:")
    print("(Press Enter to use default values)")

    try:
        u_mass = float(input("Mass uncertainty (g) [default: 0.01]: ") or 0.01)
        u_brix = float(input("Brix uncertainty (%) [default: 0.5]: ") or 0.5)
    except ValueError:
        print("Invalid input. Using defaults.")
        u_mass = 0.01
        u_brix = 0.5

    brix_factor = CONFIG["brix_to_tds_factor"]

    results = []
    for idx, row in all_data.iterrows():
        dry_mass = row["Dry Coffee Mass (g)"]
        bev_mass = row["Beverage Mass (g)"]
        brix = row["Brix"]
        tds = row["TDS"]
        yield_val = row["Extraction Yield (%)"]

        # TDS uncertainty: u_TDS = (brix_factor/100) * u_brix
        u_tds = (brix_factor / 100) * u_brix

        # Extraction yield: Y = (TDS * bev_mass / dry_mass) * 100
        # Using partial derivatives for uncertainty propagation
        partial_tds = (bev_mass / dry_mass) * 100
        partial_bev = (tds / dry_mass) * 100
        partial_dry = -(tds * bev_mass / (dry_mass**2)) * 100

        # Combined uncertainty
        u_yield = np.sqrt(
            (partial_tds * u_tds)**2 +
            (partial_bev * u_mass)**2 +
            (partial_dry * u_mass)**2
        )

        results.append({
            "Extraction": idx + 1,
            "Yield (%)": yield_val,
            "Uncertainty (%)": u_yield,
            "Relative Uncertainty (%)": (u_yield / yield_val) * 100 if yield_val != 0 else 0
        })

    results_df = pd.DataFrame(results)

    print(f"\n📊 Uncertainty Analysis Results:")
    print(f"   Input Uncertainties:")
    print(f"   - Mass: ±{u_mass} g")
    print(f"   - Brix: ±{u_brix}%")
    print(f"   - TDS: ±{u_tds:.4f}")
    print(f"\n   Extraction Yield Uncertainties:")
    print(results_df.to_string(index=False))

    avg_uncertainty = results_df["Uncertainty (%)"].mean()
    avg_rel_uncertainty = results_df["Relative Uncertainty (%)"].mean()
    print(f"\n   Average absolute uncertainty: ±{avg_uncertainty:.3f}%")
    print(f"   Average relative uncertainty: ±{avg_rel_uncertainty:.2f}%")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar plot with error bars
    x = results_df["Extraction"]
    y = results_df["Yield (%)"]
    yerr = results_df["Uncertainty (%)"]

    axes[0].bar(x, y, color='skyblue', edgecolor='black', linewidth=0.5, alpha=0.7)
    axes[0].errorbar(x, y, yerr=yerr, fmt='none', ecolor='red',
                    capsize=5, capthick=2, label='±1σ Uncertainty')
    axes[0].set_xlabel('Extraction Number', fontsize=12)
    axes[0].set_ylabel('Extraction Yield (%)', fontsize=12)
    axes[0].set_title('Extraction Yield with Uncertainties', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Relative uncertainty plot
    axes[1].plot(x, results_df["Relative Uncertainty (%)"],
                marker='o', markersize=8, linewidth=2, color='coral')
    axes[1].axhline(y=avg_rel_uncertainty, color='red', linestyle='--',
                   label=f'Average: {avg_rel_uncertainty:.2f}%')
    axes[1].set_xlabel('Extraction Number', fontsize=12)
    axes[1].set_ylabel('Relative Uncertainty (%)', fontsize=12)
    axes[1].set_title('Relative Uncertainty Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    save_plot = input("\nDo you want to save the uncertainty plot? (y/n): ").strip().lower()
    if save_plot == "y":
        figures_dir = os.path.join(DATA_DIR, "figures")
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(figures_dir, f"uncertainty_analysis_{timestamp}.png")
        plt.savefig(plot_filename, dpi=CONFIG["visualization"]["figure_dpi"], bbox_inches='tight')
        print(f"✓ Uncertainty plot saved as {plot_filename}")

    plt.show()


def advanced_visualizations() -> None:
    """
    Create advanced visualizations: box plots, violin plots, and correlation heatmap.
    """
    all_data = load_all_data()
    if all_data is None:
        return

    print("\n" + "="*60)
    print("  📈 Advanced Visualizations")
    print("="*60)
    print("\nSelect visualization type:")
    print("  1. Box Plot Comparison")
    print("  2. Violin Plot Distribution")
    print("  3. Correlation Heatmap")
    print("  4. All of the above")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice in ["1", "4"]:
        create_box_plots(all_data)

    if choice in ["2", "4"]:
        create_violin_plots(all_data)

    if choice in ["3", "4"]:
        create_correlation_heatmap(all_data)


def create_box_plots(all_data: pd.DataFrame) -> None:
    """Create box plots for different parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Parameter Distributions - Box Plots', fontsize=16, fontweight='bold')

    # Temperature
    sns.boxplot(data=all_data, y='Temperature (°C)', ax=axes[0, 0], color='lightcoral')
    axes[0, 0].set_title('Temperature Distribution', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Grind Size
    grind_order = ['fine', 'medium', 'coarse']
    grind_data = all_data.dropna(subset=['Grind Size'])
    if not grind_data.empty:
        sns.boxplot(data=grind_data, x='Grind Size', y='Extraction Yield (%)',
                   order=grind_order, ax=axes[0, 1], palette='Set2')
        axes[0, 1].set_title('Yield by Grind Size', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Extraction Time
    sns.boxplot(data=all_data, y='Extraction Time (s)', ax=axes[1, 0], color='lightblue')
    axes[1, 0].set_title('Extraction Time Distribution', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Extraction Yield
    sns.boxplot(data=all_data, y='Extraction Yield (%)', ax=axes[1, 1], color='lightgreen')
    axes[1, 1].set_title('Extraction Yield Distribution', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_plot = input("\nDo you want to save the box plots? (y/n): ").strip().lower()
    if save_plot == "y":
        figures_dir = os.path.join(DATA_DIR, "figures")
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(figures_dir, f"box_plots_{timestamp}.png")
        plt.savefig(plot_filename, dpi=CONFIG["visualization"]["figure_dpi"], bbox_inches='tight')
        print(f"✓ Box plots saved as {plot_filename}")

    plt.show()


def create_violin_plots(all_data: pd.DataFrame) -> None:
    """Create violin plots for yield distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Extraction Yield Distributions - Violin Plots', fontsize=16, fontweight='bold')

    # By Grind Size
    grind_order = ['fine', 'medium', 'coarse']
    grind_data = all_data.dropna(subset=['Grind Size'])
    if not grind_data.empty:
        sns.violinplot(data=grind_data, x='Grind Size', y='Extraction Yield (%)',
                      order=grind_order, ax=axes[0], palette='muted', inner='box')
        axes[0].set_title('Yield Distribution by Grind Size', fontsize=12)
        axes[0].grid(True, alpha=0.3, axis='y')

    # Temperature bins
    all_data['Temp Bin'] = pd.cut(all_data['Temperature (°C)'],
                                   bins=[0, 91, 93, 95, 100],
                                   labels=['<91°C', '91-93°C', '93-95°C', '>95°C'])
    temp_data = all_data.dropna(subset=['Temp Bin'])
    if not temp_data.empty:
        sns.violinplot(data=temp_data, x='Temp Bin', y='Extraction Yield (%)',
                      ax=axes[1], palette='coolwarm', inner='box')
        axes[1].set_title('Yield Distribution by Temperature Range', fontsize=12)
        axes[1].set_xlabel('Temperature Range', fontsize=11)
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_plot = input("\nDo you want to save the violin plots? (y/n): ").strip().lower()
    if save_plot == "y":
        figures_dir = os.path.join(DATA_DIR, "figures")
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(figures_dir, f"violin_plots_{timestamp}.png")
        plt.savefig(plot_filename, dpi=CONFIG["visualization"]["figure_dpi"], bbox_inches='tight')
        print(f"✓ Violin plots saved as {plot_filename}")

    plt.show()


def create_correlation_heatmap(all_data: pd.DataFrame) -> None:
    """Create correlation heatmap for numeric variables."""
    numeric_cols = ['Dry Coffee Mass (g)', 'Temperature (°C)', 'Pressure (bar)',
                   'Extraction Time (s)', 'Grind Size Numeric', 'Beverage Mass (g)',
                   'Brix', 'TDS', 'Extraction Yield (%)']

    # Filter to only existing numeric columns
    available_cols = [col for col in numeric_cols if col in all_data.columns]
    corr_data = all_data[available_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix - Espresso Extraction Parameters',
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    save_plot = input("\nDo you want to save the correlation heatmap? (y/n): ").strip().lower()
    if save_plot == "y":
        figures_dir = os.path.join(DATA_DIR, "figures")
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(figures_dir, f"correlation_heatmap_{timestamp}.png")
        plt.savefig(plot_filename, dpi=CONFIG["visualization"]["figure_dpi"], bbox_inches='tight')
        print(f"✓ Correlation heatmap saved as {plot_filename}")

    plt.show()


def main() -> None:
    """
    Main menu for choosing between data collection and visualization options.

    Supports both interactive menu mode and command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Espresso Extraction Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python espresso_extraction.py                 # Interactive menu
  python espresso_extraction.py --collect       # Collect data directly
  python espresso_extraction.py --plot-2d       # Create 2D plot
  python espresso_extraction.py --plot-3d       # Create 3D plot
  python espresso_extraction.py --export        # Export data to Excel/JSON
  python espresso_extraction.py --outliers      # Detect outliers
  python espresso_extraction.py --uncertainty   # Uncertainty analysis
  python espresso_extraction.py --advanced      # Advanced visualizations
        """
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect espresso extraction data"
    )
    parser.add_argument(
        "--plot-2d",
        action="store_true",
        help="Create 2D visualization"
    )
    parser.add_argument(
        "--plot-3d",
        action="store_true",
        help="Create 3D visualization"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export data to Excel and JSON"
    )
    parser.add_argument(
        "--outliers",
        action="store_true",
        help="Detect outliers in data"
    )
    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="Calculate uncertainty propagation"
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Create advanced visualizations"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        help=f"Directory for data files (default: {DATA_DIR})"
    )

    args = parser.parse_args()

    # Update global DATA_DIR if specified
    if args.data_dir != DATA_DIR:
        globals()['DATA_DIR'] = args.data_dir

    # Handle command-line arguments
    if args.collect:
        collect_data()
        return
    elif args.plot_2d:
        visualize_data_2d()
        return
    elif args.plot_3d:
        visualize_data_3d()
        return
    elif args.export:
        export_data()
        return
    elif args.outliers:
        detect_outliers()
        return
    elif args.uncertainty:
        calculate_uncertainty()
        return
    elif args.advanced:
        advanced_visualizations()
        return

    # Interactive menu mode
    print("\n" + "="*60)
    print("  ☕ Espresso Extraction Analysis Tool v2.1 ☕")
    print("="*60)
    print("\nA metrology-driven approach to perfecting espresso")
    print("\n📊 Data Collection & Visualization:")
    print("  1. Collect Data")
    print("  2. Visualize 2D Plot")
    print("  3. Visualize 3D Surface Plot")
    print("\n🔬 Analysis & Export:")
    print("  4. Export Data (Excel/JSON)")
    print("  5. Detect Outliers")
    print("  6. Uncertainty Propagation")
    print("  7. Advanced Visualizations")
    print("\n  8. Exit")
    print("-"*60)

    choice = input("\nEnter your choice (1-8): ").strip()

    if choice == "1":
        collect_data()
    elif choice == "2":
        visualize_data_2d()
    elif choice == "3":
        visualize_data_3d()
    elif choice == "4":
        export_data()
    elif choice == "5":
        detect_outliers()
    elif choice == "6":
        calculate_uncertainty()
    elif choice == "7":
        advanced_visualizations()
    elif choice == "8":
        print("\nExiting. Happy brewing! ☕")
        return
    else:
        print("\n⚠ Invalid choice. Please enter 1-8.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user. Exiting...")
    except Exception as e:
        print(f"\n⚠ An error occurred: {e}")
        import traceback
        traceback.print_exc()
