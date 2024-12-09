# Back Biomechanical Analysis
# This file analyzes spine oscillations and compares two different settings.
# It provides various graphs for visualizing oscillations:
# - Lateral oscillations of the spine
# - Envelope of oscillations for maximum amplitude trends
# - oscillations of a point of the spine (ab point)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_spine_data, calculate_angle
from scipy.signal import hilbert

# Analyze spine oscillations
def analyze_spine_oscillations(df):
    hip_positions = df[["hip_x", "hip_y"]].values
    chest_positions = df[["chest_x", "chest_y"]].values

    mean_spine_vector = np.mean(chest_positions - hip_positions, axis=0)
    mean_spine_vector /= np.linalg.norm(mean_spine_vector)

    oscillations = []
    for frame in range(len(hip_positions)):
        current_spine_vector = chest_positions[frame] - hip_positions[frame]
        lateral_deviation = np.cross(mean_spine_vector, current_spine_vector)
        oscillations.append(lateral_deviation)

    oscillations = np.array(oscillations)

    summary_stats = {
        "mean_oscillation": np.mean(oscillations),
        "std_oscillation": np.std(oscillations),
        "max_oscillation": np.max(np.abs(oscillations)),
    }

    return pd.Series(oscillations, name="oscillations"), summary_stats

# Plot spine oscillations
def plot_spine_oscillations(oscillations, title="Spine Oscillations Over Time"):
    plt.figure(figsize=(10, 6))
    plt.plot(oscillations, label="Lateral Oscillations", color="blue")
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Lateral Deviation")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8, label="Mean Line")
    plt.legend()
    plt.grid()
    plt.show()

# REMOVE? Plot combined spine oscillations
def plot_combined_spine_oscillations(oscillations1, oscillations2, title="Ab Spine Oscillations Comparison"):
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

# Envelope of oscillations
def plot_envelope(oscillations, title="Envelope of Oscillations"):
    """
    Confronto tra settings: Se un setting ha un envelope più costante, può essere indice di maggiore stabilità del movimento.
    Analisi di stabilità: Oscillazioni molto variabili (con envelope che fluttua molto) possono indicare mancanza di controllo o irregolarità.
    Energia del movimento: Se l'envelope cresce o diminuisce nel tempo, può indicare cambiamenti nell'intensità del movimento.
    """
    analytic_signal = hilbert(oscillations)
    amplitude_envelope = np.abs(analytic_signal)

    plt.figure(figsize=(10, 6))
    plt.plot(oscillations, label="Oscillations", color="blue", lw=0.8, alpha=0.4)
    plt.plot(amplitude_envelope, label="Envelope", color="blue", lw=1.2 )
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Lateral Deviation")
    plt.legend()
    plt.grid()
    plt.show()

# Calculate 'ab' point statistics and oscillations
def calculate_ab_statistics(df):
    ab_x, ab_y, ab_z = df["ab_x"], df["ab_y"], df["ab_z"]

    # Oscillations
    oscillations = df["ab_y"] - df["ab_y"].mean()

    # Calculate differences to estimate velocities
    velocities = np.sqrt(np.diff(ab_x)**2 + np.diff(ab_y)**2 + np.diff(ab_z)**2)
    mean_velocity = np.mean(velocities)
    std_velocity = np.std(velocities)
    
    stats = {
        "mean_x": ab_x.mean(),
        "std_x": ab_x.std(),
        "range_x": ab_x.max() - ab_x.min(),
        "mean_y": ab_y.mean(),
        "std_y": ab_y.std(),
        "range_y": ab_y.max() - ab_y.min(),
        "mean_z": ab_z.mean(),
        "std_z": ab_z.std(),
        "range_z": ab_z.max() - ab_z.min(),
        "mean_velocity": mean_velocity,
        "std_velocity": std_velocity,
    }
    
    return pd.Series(oscillations, name="ab_oscillations"), stats


#SPINE ALIGNMENT TO THE GROUND
def calculate_spine_alignment(df):
    """
    Calculate alignment angles of the spine segments (hip-ab and ab-chest) and relative to the ground.

    Parameters:
        df (pd.DataFrame): DataFrame containing spine joint positions.

    Returns:
        pd.DataFrame: Angles between spine segments and with the ground over time.
    """
    angles = []

    for idx, row in df.iterrows():
        hip = np.array([row["hip_x"], row["hip_y"], row["hip_z"]])
        ab = np.array([row["ab_x"], row["ab_y"], row["ab_z"]])
        chest = np.array([row["chest_x"], row["chest_y"], row["chest_z"]])

        # Vectors for spine segments
        hip_ab = ab - hip
        ab_chest = chest - ab

        # Angle between hip-ab and ab-chest
        angle_segments = calculate_angle(hip_ab, ab_chest)

        # Angle of hip-chest relative to the ground (z-axis)
        hip_chest = chest - hip
        ground_vector = np.array([0, 0, 1])
        angle_ground = calculate_angle(hip_chest, ground_vector)

        angles.append({
            "frame": idx,
            "angle_segments": angle_segments,
            "angle_ground": angle_ground
        })

    return pd.DataFrame(angles)

def plot_spine_alignment(angles_df, title="Spine Alignment Angles"):
    """
    Plot spine alignment angles over time.

    Parameters:
        angles_df (pd.DataFrame): DataFrame containing alignment angles over time.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))

    # Plot angle between segments
    plt.plot(angles_df["frame"], angles_df["angle_segments"], label="Angle Between Segments", color="blue")

    # Plot angle relative to ground
    plt.plot(angles_df["frame"], angles_df["angle_ground"], label="Angle Relative to Ground", color="green")

    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.grid()
    plt.show()

# Plot combined spine alignment angles
def plot_combined_spine_alignment(angles_df1, angles_df2, title="Combined Spine Alignment Angles"):
    """
    Plot combined spine alignment angles for two settings.

    Parameters:
        angles_df1 (pd.DataFrame): Alignment angles for setting 1.
        angles_df2 (pd.DataFrame): Alignment angles for setting 2.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))

    # Plot angle relative to ground
    plt.plot(angles_df1["frame"], angles_df1["angle_ground"], label="Ground - Setting 1", color="red")
    plt.plot(angles_df2["frame"], angles_df2["angle_ground"], label="Ground - Setting 2", color="blue")

    # Plot angle between segments
    plt.plot(angles_df1["frame"], angles_df1["angle_segments"], label="Segments - Setting 1", color="darkred")
    plt.plot(angles_df2["frame"], angles_df2["angle_segments"], label="Segments - Setting 2", color="darkblue")

    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.grid()
    plt.show()

# Load files
file_1 = "output/spine_metrics_1.json"
file_2 = "output/spine_metrics_2.json"

df_1 = load_spine_data(file_1)
df_2 = load_spine_data(file_2)

# Analyze spine oscillations
# oscillations_1, stats_1 = analyze_spine_oscillations(df_1)
# oscillations_2, stats_2 = analyze_spine_oscillations(df_2)
# ab_oscillations_1, stats_ab_1 = calculate_ab_statistics(df_1)
# ab_oscillations_2, stats_ab_2 = calculate_ab_statistics(df_2)
alignment_angles_1 = calculate_spine_alignment(df_1)
alignment_angles_2 = calculate_spine_alignment(df_2)

# #statistics about oscillation of the back
# print("Setting 1 - Oscillation Statistics:", stats_1)
# print("Setting 2 - Oscillation Statistics:", stats_2)

# # Print 'ab' statistics
# print("\nStatistics for 'ab' Point - Setting 1:")
# for stat, value in stats_ab_1.items():
#     print(f"  {stat}: {value:.2f}")

# print("\nStatistics for 'ab' Point - Setting 2:")
# for stat, value in stats_ab_2.items():
#     print(f"  {stat}: {value:.2f}")

# Generate plots
# plot_spine_oscillations(oscillations_1, title="Spine Oscillations - Setting 1")
# plot_spine_oscillations(oscillations_2, title="Spine Oscillations - Setting 2")
# # plot_combined_spine_oscillations(oscillations_1, oscillations_2)
# plot_envelope(ab_oscillations_1, title="Envelope-1 - 'ab' Point")
# plot_envelope(ab_oscillations_2, title="Envelope-2 - 'ab' Point")
# plot_spine_alignment(alignment_angles_1, title="Spine Alignment Angles - Setting 1")
# plot_spine_alignment(alignment_angles_2, title="Spine Alignment Angles - Setting 2")
plot_combined_spine_alignment(alignment_angles_1, alignment_angles_2, title="Combined Spine Alignment Angles")



