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
  python espresso_extraction.py              # Interactive menu
  python espresso_extraction.py --collect    # Collect data directly
  python espresso_extraction.py --plot-2d    # Create 2D plot
  python espresso_extraction.py --plot-3d    # Create 3D plot
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

    # Interactive menu mode
    print("\n" + "="*50)
    print("  ☕ Espresso Extraction Analysis Tool ☕")
    print("="*50)
    print("\nA metrology-driven approach to perfecting espresso")
    print("\nOptions:")
    print("  1. Collect Data")
    print("  2. Visualize 2D Plot")
    print("  3. Visualize 3D Surface Plot")
    print("  4. Exit")
    print("-"*50)

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        collect_data()
    elif choice == "2":
        visualize_data_2d()
    elif choice == "3":
        visualize_data_3d()
    elif choice == "4":
        print("\nExiting. Happy brewing! ☕")
        return
    else:
        print("\n⚠ Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user. Exiting...")
    except Exception as e:
        print(f"\n⚠ An error occurred: {e}")
        import traceback
        traceback.print_exc()
