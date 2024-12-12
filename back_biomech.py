import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_spine_data

def analyze_spine_oscillations(df):
    """
    Analyze lateral oscillations of the spine over time.

    Parameters:
        df (pd.DataFrame): DataFrame containing spine joint positions.
    
    Returns:
        pd.Series, dict: Oscillations over time and summary statistics.
    """
    # Extract x, y positions of hip and chest
    hip_positions = df[["hip_x", "hip_y"]].values
    chest_positions = df[["chest_x", "chest_y"]].values

    # Calculate the mean direction of the spine (hip → chest) over all frames
    mean_spine_vector = np.mean(chest_positions - hip_positions, axis=0)
    mean_spine_vector /= np.linalg.norm(mean_spine_vector)  # Normalize

    # Calculate lateral oscillations
    oscillations = []
    for frame in range(len(hip_positions)):
        # Current spine vector
        current_spine_vector = chest_positions[frame] - hip_positions[frame]
        # Lateral deviation (orthogonal to the mean vector)
        lateral_deviation = np.cross(mean_spine_vector, current_spine_vector)
        oscillations.append(lateral_deviation)

    oscillations = np.array(oscillations)

    # Summary statistics
    mean_oscillation = np.mean(oscillations)
    std_oscillation = np.std(oscillations)
    max_oscillation = np.max(np.abs(oscillations))

    summary_stats = {
        "mean_oscillation": mean_oscillation,
        "std_oscillation": std_oscillation,
        "max_oscillation": max_oscillation,
    }

    return pd.Series(oscillations, name="oscillations"), summary_stats

def plot_spine_oscillations(oscillations, title="Spine Oscillations Over Time"):
    """
    Plot lateral spine oscillations over time.

    Parameters:
        oscillations (pd.Series): Oscillations over time.
        title (str): Title of the plot.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(oscillations, label="Lateral Oscillations", color="blue")
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Lateral Deviation")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8, label="Mean Line")
    plt.legend()
    plt.grid()
    plt.show()

def calculate_ab_oscillations(df):
    """
    Calculate the oscillations of the 'ab' point (midpoint of the spine) relative to the y-axis.

    Parameters:
        df (pd.DataFrame): DataFrame containing spine joint positions.

    Returns:
        pd.Series: 'ab' point oscillations over time.
    """
    # Extract the y-coordinate of the 'ab' point
    ab_y = df["ab_y"]

    # Calculate mean position along the y-axis
    mean_ab_y = ab_y.mean()

    # Calculate oscillations (deviation from mean)
    oscillations = ab_y - mean_ab_y

    return pd.Series(oscillations, name="ab_oscillations")

def plot_combined_spine_oscillations(oscillations1, oscillations2, title="Ab spine Oscillations Comparison"):
    """
    Plot lateral spine oscillations for two settings in a single graph.

    Parameters:
        oscillations1 (pd.Series): Oscillations for setting 1.
        oscillations2 (pd.Series): Oscillations for setting 2.
        title (str): Title of the plot.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(oscillations1, label="Setting 1", color="blue")
    plt.plot(oscillations2, label="Setting 2", color="green")
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Lateral Deviation")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8, label="Mean Line")
    plt.legend()
    plt.grid()
    plt.show()



# Load files
file_1 = "output/spine_metrics_1.json"
file_2 = "output/spine_metrics_2.json"

df1 = load_spine_data(file_1)
df2 = load_spine_data(file_2)

# Analyze spine oscillations
oscillations1, stats1 = analyze_spine_oscillations(df1)
oscillations2, stats2 = analyze_spine_oscillations(df2)

# Calculate oscillations for 'ab'
ab_oscillations1 = calculate_ab_oscillations(df1)
ab_oscillations2 = calculate_ab_oscillations(df2)

# Print summary statistics
print("Setting 1 - Oscillation Statistics:", stats1)
print("Setting 2 - Oscillation Statistics:", stats2)

# Plot oscillations
plot_spine_oscillations(oscillations1, title="Spine Oscillations - Setting 1")
plot_spine_oscillations(oscillations2, title="Spine Oscillations - Setting 2")

plot_combined_spine_oscillations(ab_oscillations1, ab_oscillations2, title="'ab' Point Oscillations Comparison")

