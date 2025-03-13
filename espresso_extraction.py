import os
import csv
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from scipy.interpolate import griddata  # For surface interpolation

# Directory for storing data files
DATA_DIR = "extraction_data"


def get_input_with_default(prompt, default, cast_func=str):
    """
    Prompt the user for input with a default.
    Returns a tuple: (value, flag) where flag is True if the user provided a value.
    """
    user_input = input(f"{prompt} (default: {default}): ")
    if user_input.strip() == "":
        return default, False
    else:
        try:
            return cast_func(user_input), True
        except Exception as e:
            print(f"Invalid input. Using default {default}.")
            return default, False


def collect_data():
    """Collect espresso extraction trial data in the specified order and save it as a CSV file."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1) Dry coffee weight
    dry_coffee_mass = float(input("Enter weight of dry coffee grounds (g): "))

    # 2) Temperature
    temperature, temp_used = get_input_with_default("Enter extraction temperature (°C)", 93, float)

    # 3) Pressure
    pressure, pressure_used = get_input_with_default("Enter extraction pressure (bar)", 9, float)

    # 4) Grind size
    grind_size, grind_used = get_input_with_default("Enter grind size (fine, medium, coarse)", "medium", str)

    # 5) Extraction time
    extraction_time, time_used = get_input_with_default("Enter extraction time (s)", 25, float)

    # 6) Weight of pulled shot
    beverage_mass = float(input("Enter weight of pulled shot (g): "))

    # 7) Brix
    brix = float(input("Enter Brix reading (as a percentage): "))

    # Calculate TDS and extraction yield
    # Brix is treated as a percent, so 25 => 25%. Convert to fraction, then apply factor 0.85.
    tds = (brix * 0.85) / 100
    extraction_yield = (tds * beverage_mass) / dry_coffee_mass * 100

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

    print(f"Data saved to {filename}")


def visualize_data_2d():
    """Load all extraction trial data and plot extraction yield vs. chosen metadata (2D scatter plot)."""
    files = glob.glob(os.path.join(DATA_DIR, "extraction_*.csv"))
    if not files:
        print("No data files found in extraction_data directory.")
        return

    all_data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Map grind size to micron values: fine=200, medium=400, coarse=600
    grind_size_map = {"fine": 200, "medium": 400, "coarse": 600}
    if "Grind Size" in all_data.columns:
        all_data["Grind Size Numeric"] = all_data["Grind Size"].str.lower().map(grind_size_map)

    print("Select metadata for the x-axis (2D plot):")
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

    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, alpha=0.7, label='Experimental Data')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # New, less literal title
    plt.title(f"{x_label}: Impact on Extraction Yield")
    plt.grid(True)

    # Fit a linear regression trend line using least squares
    coeffs = np.polyfit(x_data, y_data, 1)
    m, c = coeffs
    poly = np.poly1d(coeffs)
    # Reduced sampling: 50 phantom points
    x_fit = np.linspace(x_data.min(), x_data.max(), 50)
    y_fit = poly(x_fit)
    # Plot phantom trend points in light blue
    plt.plot(x_fit, y_fit, color='lightblue', linestyle='-', marker='o', markersize=4, label='Trend (Interpolated)')
    # Annotate the trend equation and method
    eq_text = f"Trend: y = {m:.2f}x + {c:.2f}\n(Least Squares Regression)"
    plt.annotate(eq_text, xy=(0.65, 0.15), xycoords='axes fraction', fontsize=10,
                 horizontalalignment='left', verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))
    plt.legend()

    save_plot = input("Do you want to save the 2D plot? (y/n): ").strip().lower()
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
        plt.savefig(plot_filename, dpi=600)
        print(f"2D plot saved as {plot_filename}")

    plt.show()


def visualize_data_3d():
    """
    Load all extraction trial data and create a 3D scatter plot of Extraction Yield (%)
    vs. two chosen metadata fields. A trend plane is fitted using multiple linear regression.
    The real (experimental) data points are shown in red.
    """
    files = glob.glob(os.path.join(DATA_DIR, "extraction_*.csv"))
    if not files:
        print("No data files found in extraction_data directory.")
        return

    all_data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Map grind size to micron values: fine=200, medium=400, coarse=600
    grind_size_map = {"fine": 200, "medium": 400, "coarse": 600}
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

    print("Select metadata for the x-axis (3D plot):")
    for key, (label, _) in options.items():
        print(f"{key}: {label}")
    x_choice = input("Enter your choice for x-axis (1-5): ").strip()
    if x_choice not in options:
        print("Invalid choice. Defaulting x-axis to Temperature (°C).")
        x_choice = "2"
    x_label, x_data = options[x_choice]

    print("Select metadata for the y-axis (3D plot):")
    for key, (label, _) in options.items():
        print(f"{key}: {label}")
    y_choice = input("Enter your choice for y-axis (1-5): ").strip()
    if y_choice not in options:
        print("Invalid choice. Defaulting y-axis to Pressure (bar).")
        y_choice = "3"
    y_label, y_data = options[y_choice]

    # z-axis is now Extraction Yield (%)
    z_label = "Extraction Yield (%)"
    z_data = all_data[z_label]

    # Convert to numpy arrays for regression
    x_array = np.array(x_data)
    y_array = np.array(y_data)
    z_array = np.array(z_data)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot real data in red
    ax.scatter(x_array, y_array, z_array, color='red', s=40, marker='o', label='Experimental Data')

    # Fit a plane using multiple linear regression: z = a*x + b*y + c
    A = np.column_stack((x_array, y_array, np.ones(len(x_array))))
    coeffs, residuals, rank, s = np.linalg.lstsq(A, z_array, rcond=None)
    a, b, c = coeffs
    # New, less literal title
    ax.set_title("Charting the Coffee Extraction Landscape", pad=20)
    # Include the trend plane details as a subtitle in the legend
    trend_text = f"Trend Plane (Least Squares): y = {a:.2f}x + {b:.2f}y + {c:.2f}"

    # Generate phantom trend points with a reduced sampling rate (15 points)
    x_fit = np.linspace(x_array.min(), x_array.max(), 15)
    y_fit = np.linspace(y_array.min(), y_array.max(), 15)
    x_fit_grid, y_fit_grid = np.meshgrid(x_fit, y_fit)
    z_fit = a * x_fit_grid + b * y_fit_grid + c
    # Overlay phantom trend points in light blue
    ax.scatter(x_fit_grid, y_fit_grid, z_fit, color='lightblue', s=40, marker='^', label=trend_text)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.legend()

    save_plot = input("Do you want to save the 3D plot? (y/n): ").strip().lower()
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
        plt.savefig(plot_filename, dpi=600)
        print(f"3D plot saved as {plot_filename}")

    plt.show()


def main():
    """Main menu for choosing between data collection and data visualization options."""
    print("Espresso Extraction Analysis")
    print("1. Collect Data")
    print("2. Visualize 2D Plot")
    print("3. Visualize 3D Surface Plot")
    choice = input("Enter your choice (1, 2, or 3): ").strip()

    if choice == "1":
        collect_data()
    elif choice == "2":
        visualize_data_2d()
    elif choice == "3":
        visualize_data_3d()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
