"""
Leg Biomechanical Analysis
This file focuses on the biomechanical analysis of lower limb angles during pedaling, comparing two different settings. 

Key Analyses:
- Extract and analyze knee and ankle joint angles over time.
- Compare metrics such as mean, standard deviation, and amplitude of cycles between settings.
- Visualize the results through various graphs:
  1. Cycle Amplitudes Comparison
  2. Detailed Cycle Analysis (Angle and Angular Velocity)
  3. Distribution of anglees (peaks and valleys)
  4. Comparison between angles with polar graph (peaks and valleys) 
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Function to analyze a single angle
def analyze_knee_angle(file_path, angle="knee_r"):
    """
    Analyze a specific angle from a JSON file.
    """
    # Load JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    angles = df[angle].values

    # peaks (maxima) and valleys (minima). CHECK PROMINENCE
    peaks, _ = find_peaks(angles, prominence=5)  
    valleys, _ = find_peaks(-angles, prominence=5)

    # equal number of peaks and valleys to extract cycles
    num_cycles = min(len(peaks), len(valleys))
    peaks = peaks[:num_cycles]
    valleys = valleys[:num_cycles]

    # cycle metrics
    cycle_amplitudes = angles[peaks] - angles[valleys]
    cycle_durations = np.diff(peaks)
    cadence = 60 / (np.mean(cycle_durations) / 30)  # 30 FPS

    # angular velocity
    angular_velocity = np.gradient(angles, 1 / 30)  # 30 FPS
    angular_velocity_mean = np.mean(angular_velocity)
    angular_velocity_std = np.std(angular_velocity)

    stats = {
        "mean": np.mean(angles),
        "std": np.std(angles),
        "min": np.min(angles),
        "max": np.max(angles),
        "range": np.max(angles) - np.min(angles),
        "cycle_amplitude_mean": np.mean(cycle_amplitudes),
        "cycle_amplitude_std": np.std(cycle_amplitudes),
        "cadence": cadence,
        "angular_velocity_mean": angular_velocity_mean,
        "angular_velocity_std": angular_velocity_std
    }

    return stats, peaks, valleys, angles, cycle_amplitudes, cycle_durations

def compare_settings(file_1, file_2, angles):
    """
    Compare statistics for specific angles between two settings.
    """
    comparison_output = {}
    for angle in angles:
        stats1, _, _, _, _, _ = analyze_knee_angle(file_1, angle=angle)
        stats2, _, _, _, _, _ = analyze_knee_angle(file_2, angle=angle)

        comparison_output[angle.upper()] = {
            "mean": f"Before: {stats1['mean']:.2f} - After: {stats2['mean']:.2f}",
            "std": f"Before: {stats1['std']:.2f} - After: {stats2['std']:.2f}",
            "min": f"Before: {stats1['min']:.2f} - After: {stats2['min']:.2f}",
            "max": f"Before: {stats1['max']:.2f} - After: {stats2['max']:.2f}",
            "range": f"Before: {stats1['range']:.2f} - After: {stats2['range']:.2f}",
            "cycle_amplitude_mean": f"Before: {stats1['cycle_amplitude_mean']:.2f} - After: {stats2['cycle_amplitude_mean']:.2f}",
            "cycle_amplitude_std": f"Before: {stats1['cycle_amplitude_std']:.2f} - After: {stats2['cycle_amplitude_std']:.2f}",
            "cadence": f"Before: {stats1['cadence']:.2f} - After: {stats2['cadence']:.2f}",
            "angular_velocity_mean": f"Before: {stats1['angular_velocity_mean']:.2f} - After: {stats2['angular_velocity_mean']:.2f}",
            "angular_velocity_std": f"Before: {stats1['angular_velocity_std']:.2f} - After: {stats2['angular_velocity_std']:.2f}"
        }

    return comparison_output

def calculate_angular_velocity(angles, frame_rate=30):
    """
    Calculate angular velocity for a given angle series.
    """
    return np.gradient(angles, 1 / frame_rate)

def plot_cycle_amplitudes(angles1, angles2, peaks1, peaks2, valleys1, valleys2):
    """
    Plot cycle amplitudes for both settings.
    """
    # Calculate amplitudes for each cycle
    cycle_amplitudes1 = angles1[peaks1] - angles1[valleys1[:len(peaks1)]]
    cycle_amplitudes2 = angles2[peaks2] - angles2[valleys2[:len(peaks2)]]

    plt.figure(figsize=(10, 6))
    plt.plot(cycle_amplitudes1, label="Setting 1 (Amplitude)", marker="o", linestyle="--", color="blue")
    plt.plot(cycle_amplitudes2, label="Setting 2 (Amplitude)", marker="o", linestyle="-", color="green")
    plt.title("Cycle Amplitudes Comparison")
    plt.xlabel("Cycle Number")
    plt.ylabel("Amplitude (degrees)")
    plt.ylim(60, 100)
    plt.legend()
    plt.grid()
    plt.show()

def plot_angle_and_velocity_for_cycle(angles1, angles2, peaks1, peaks2, cycle_index, frame_rate=30):
    """
    Plot angle and angular velocity for a specific cycle for two settings.
    """
    # Extract cycle data for Setting 1
    start_frame1 = peaks1[cycle_index]
    end_frame1 = peaks1[cycle_index + 1]
    angles_cycle1 = angles1[start_frame1:end_frame1]
    velocity_cycle1 = np.gradient(angles_cycle1, 1 / frame_rate)

    # Extract cycle data for Setting 2
    start_frame2 = peaks2[cycle_index]
    end_frame2 = peaks2[cycle_index + 1]
    angles_cycle2 = angles2[start_frame2:end_frame2]
    velocity_cycle2 = np.gradient(angles_cycle2, 1 / frame_rate)

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot angles on primary Y-axis
    ax1.plot(angles_cycle1, label="Setting 1: Angle (degrees)", color="red", linestyle="-")
    ax1.plot(angles_cycle2, label="Setting 2: Angle (degrees)", color="blue", linestyle="-")
    ax1.set_xlabel("Frame (within cycle)")
    ax1.set_ylabel("Angle (degrees)", color="black")
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.legend(loc="upper left")
    ax1.grid()

    # Create secondary Y-axis for angular velocity
    ax2 = ax1.twinx()
    ax2.plot(velocity_cycle1, label="Setting 1: Angular Velocity (degrees/second)", color="red", linestyle="--")
    ax2.plot(velocity_cycle2, label="Setting 2: Angular Velocity (degrees/second)", color="blue", linestyle="--")
    ax2.set_ylabel("Angular Velocity (degrees/second)", color="black")
    ax2.tick_params(axis='y', labelcolor="black")
    ax2.legend(loc="upper right")

    # Title and show plot
    plt.title(f"Angle and Angular Velocity for Cycle {cycle_index + 1}")
    plt.show()

# Scatter plot for angle distribution
def plot_angle_distribution(peaks1, valleys1, peaks2, valleys2, angles1, angles2):
    """
    Plot scatter distribution of maximum and minimum angle values and their normal distributions.
    """
    from scipy.stats import norm

    # Calculate stats for maxima (peaks)
    mean_peaks1 = np.mean(angles1[peaks1])
    std_peaks1 = np.std(angles1[peaks1])
    mean_peaks2 = np.mean(angles2[peaks2])
    std_peaks2 = np.std(angles2[peaks2])

    # Scatter plot for maxima (peaks)
    plt.figure(figsize=(10, 6))
    plt.hist(angles1[peaks1], bins=30, alpha=0.6, color="red", label="Setting 1 Peaks", density=True)
    plt.hist(angles2[peaks2], bins=30, alpha=0.6, color="blue", label="Setting 2 Peaks", density=True)

    # Overlay normal distributions
    x = np.linspace(min(angles1[peaks1].min(), angles2[peaks2].min()),
                    max(angles1[peaks1].max(), angles2[peaks2].max()), 100)
    y1 = norm.pdf(x, loc=mean_peaks1, scale=std_peaks1)
    y2 = norm.pdf(x, loc=mean_peaks2, scale=std_peaks2)
    plt.plot(x, y1, color="darkred", linestyle="--", label="Setting 1 Fit")
    plt.plot(x, y2, color="darkblue", linestyle="--", label="Setting 2 Fit")

    plt.title("Angle Distribution of Maxima (Peaks)")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.show()

    # Calculate stats for minima (valleys)
    mean_valleys1 = np.mean(angles1[valleys1])
    std_valleys1 = np.std(angles1[valleys1])
    mean_valleys2 = np.mean(angles2[valleys2])
    std_valleys2 = np.std(angles2[valleys2])

    # Scatter plot for minima (valleys)
    plt.figure(figsize=(10, 6))
    plt.hist(angles1[valleys1], bins=30, alpha=0.6, color="red", label="Setting 1 Valleys", density=True)
    plt.hist(angles2[valleys2], bins=30, alpha=0.6, color="blue", label="Setting 2 Valleys", density=True)

    # Overlay normal distributions
    x = np.linspace(min(angles1[valleys1].min(), angles2[valleys2].min()),
                    max(angles1[valleys1].max(), angles2[valleys2].max()), 100)
    y1 = norm.pdf(x, loc=mean_valleys1, scale=std_valleys1)
    y2 = norm.pdf(x, loc=mean_valleys2, scale=std_valleys2)
    plt.plot(x, y1, color="darkred", linestyle="--", label="Setting 1 Fit")
    plt.plot(x, y2, color="darkblue", linestyle="--", label="Setting 2 Fit")

    plt.title("Angle Distribution of Minima (Valleys)")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.show()

def plot_polar_angles_with_frames(peaks1, valleys1, peaks2, valleys2, angles1, angles2, title):
    """
    Plot angles in polar coordinates using frame indices as the radial coordinate.
    """
    
    # Convert angles to radians for polar plot
    peaks1_angles = np.deg2rad(angles1[peaks1])
    valleys1_angles = np.deg2rad(angles1[valleys1])
    peaks2_angles = np.deg2rad(angles2[peaks2])
    valleys2_angles = np.deg2rad(angles2[valleys2])

    # Use frame indices as the radial coordinate
    peaks1_frames = peaks1
    valleys1_frames = valleys1
    peaks2_frames = peaks2
    valleys2_frames = valleys2

    # Polar plot for maxima (peaks)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.scatter(peaks1_angles, peaks1_frames, color="red", label="Setting 1 Peaks")
    ax.scatter(peaks2_angles, peaks2_frames, color="blue", label="Setting 2 Peaks")
    plt.title(f"{title} - Max Angles")
    plt.legend()
    plt.show()

    # Polar plot for minima (valleys)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.scatter(valleys1_angles, valleys1_frames, color="red", label="Setting 1 Valleys")
    ax.scatter(valleys2_angles, valleys2_frames, color="blue", label="Setting 2 Valleys")
    plt.title(f"{title} - Min Angles")
    plt.legend()
    plt.show()

# Load files
file_1 = "output/angles.json"
file_2 = "output/angles_2.json"

with open(file_1) as f:
    data1 = json.load(f)

with open(file_2) as f:
    data2 = json.load(f)

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Compare settings
angles = ["knee_l", "knee_r", "ankle_l", "ankle_r"]
comparison_output = compare_settings(file_1, file_2, angles)

# Print results
for angle, metrics in comparison_output.items():
    print(f"{angle}:")
    for metric, values in metrics.items():
        print(f"  {metric}: {values}")
    print()

#PLOTS
stats1, peaks1, valleys1, angles1, cycle_amplitudes1, cycle_durations1 = analyze_knee_angle(file_1, angle="knee_l")
stats2, peaks2, valleys2, angles2, cycle_amplitudes2, cycle_durations2 = analyze_knee_angle(file_2, angle="knee_l")

# plot_cycle_amplitudes(angles1, angles2, peaks1, peaks2, valleys1, valleys2)
plot_angle_and_velocity_for_cycle(angles1, angles2, peaks1, peaks2, cycle_index=75)

# Generate scatter plot for angle distribution
plot_angle_distribution(peaks1, valleys1, peaks2, valleys2, angles1, angles2)
plot_polar_angles_with_frames(peaks1, valleys1, peaks2, valleys2, angles1, angles2, "Angles in time")