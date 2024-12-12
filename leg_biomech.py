import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Function to analyze a single angle
def analyze_knee_angle(file_path, angle="knee_r"):
    # Load JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Extract angle values
    angles = df[angle].values

    # Find peaks (maxima) and valleys (minima)
    peaks, _ = find_peaks(angles, prominence=5)  # Adjust prominence based on noise
    valleys, _ = find_peaks(-angles, prominence=5)

    # Ensure equal number of peaks and valleys
    num_cycles = min(len(peaks), len(valleys))
    peaks = peaks[:num_cycles]
    valleys = valleys[:num_cycles]

    # Calculate cycle metrics
    cycle_amplitudes = angles[peaks] - angles[valleys]
    cycle_durations = np.diff(peaks)  # In frames
    cadence = 60 / (np.mean(cycle_durations) / 30)  # Assuming 30 FPS

    # Calculate angular velocity
    angular_velocity = np.gradient(angles, 1 / 30)  # Assuming 30 FPS
    angular_velocity_mean = np.mean(angular_velocity)
    angular_velocity_std = np.std(angular_velocity)

    # Update stats
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

# Function to compare statistics between settings
def compare_settings(file_1, file_2, angles):
    comparison_output = {}
    for angle in angles:
        stats1, _, _, _, cycle_amplitudes1, cycle_durations1 = analyze_knee_angle(file_1, angle=angle)
        stats2, _, _, _, cycle_amplitudes2, cycle_durations2 = analyze_knee_angle(file_2, angle=angle)

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

# Plot cycle amplitudes
def plot_cycle_amplitudes(angles1, angles2, peaks1, peaks2, valleys1, valleys2):
    # Calculate amplitudes for each cycle
    cycle_amplitudes1 = angles1[peaks1] - angles1[valleys1[:len(peaks1)]]
    cycle_amplitudes2 = angles2[peaks2] - angles2[valleys2[:len(peaks2)]]

    # Plot amplitudes
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

# Plot cycle durations
def plot_cycle_durations(cycle_durations1, cycle_durations2):
    # Convert to seconds (assuming 30 FPS)
    cycle_durations1 = cycle_durations1 / 30
    cycle_durations2 = cycle_durations2 / 30

    # Plot durations
    plt.figure(figsize=(10, 6))
    plt.plot(cycle_durations1, label="Setting 1 (Cycle Duration)", marker="o", linestyle="--", color="blue")
    plt.plot(cycle_durations2, label="Setting 2 (Cycle Duration)", marker="o", linestyle="-", color="green")
    plt.title("Cycle Durations Comparison")
    plt.xlabel("Cycle Number")
    plt.ylabel("Duration (seconds)")
    plt.legend()
    plt.grid()
    plt.show()

# Plot angular velocity per cycle
def calculate_angular_velocity(angles, frame_rate=30):
    return np.gradient(angles, 1 / frame_rate)

def plot_angular_velocity_per_cycle(angles1, angles2, peaks1, peaks2):
    # Calculate angular velocities
    velocity1 = calculate_angular_velocity(angles1)
    velocity2 = calculate_angular_velocity(angles2)

    # Extract velocities for each cycle
    cycle_velocity1 = [np.mean(velocity1[peaks1[i]:peaks1[i+1]]) for i in range(len(peaks1) - 1)]
    cycle_velocity2 = [np.mean(velocity2[peaks2[i]:peaks2[i+1]]) for i in range(len(peaks2) - 1)]

    # Plot velocities
    plt.figure(figsize=(10, 6))
    plt.plot(cycle_velocity1, label="Setting 1 (Angular Velocity)", marker="o", linestyle="--", color="blue")
    plt.plot(cycle_velocity2, label="Setting 2 (Angular Velocity)", marker="o", linestyle="-", color="green")
    plt.title("Angular Velocity Per Cycle Comparison")
    plt.xlabel("Cycle Number")
    plt.ylabel("Angular Velocity (degrees/second)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_angle_and_velocity_for_cycle(angles1, angles2, peaks1, peaks2, cycle_index, frame_rate=30):
    """
    Plot angle and angular velocity for a specific cycle for two settings.
    
    Args:
        angles1: Angles for setting 1.
        angles2: Angles for setting 2.
        peaks1: Peaks (cycle delimiters) for setting 1.
        peaks2: Peaks (cycle delimiters) for setting 2.
        cycle_index: Index of the cycle to analyze.
        frame_rate: Frame rate of the recording (default 30 FPS).
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

# Print the results
for angle, metrics in comparison_output.items():
    print(f"{angle}:")
    for metric, values in metrics.items():
        print(f"  {metric}: {values}")
    print()

# Example plot
stats1, peaks1, valleys1, angles1, cycle_amplitudes1, cycle_durations1 = analyze_knee_angle(file_1, angle="knee_l")
stats2, peaks2, valleys2, angles2, cycle_amplitudes2, cycle_durations2 = analyze_knee_angle(file_2, angle="knee_l")
plot_cycle_amplitudes(angles1, angles2, peaks1, peaks2, valleys1, valleys2)
plot_cycle_durations(cycle_durations1, cycle_durations2)
plot_angular_velocity_per_cycle(angles1, angles2, peaks1, peaks2) # Rimuovere, poco significativa ?
plot_angle_and_velocity_for_cycle(angles1, angles2, peaks1, peaks2, cycle_index=75)
