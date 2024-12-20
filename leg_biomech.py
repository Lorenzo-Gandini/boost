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
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from config import ATHLETE_MOD, ATHLETE_MOD_UC

# Function to analyze a single angle
def analyze_knee_angle(file_path, angle="knee_l"):
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

def calculate_distribution(data, indices):
    """
    Calcola la distribuzione KDE e restituisce i parametri necessari per il fit.
    """
    filtered_data = data[indices]
    kde = gaussian_kde(filtered_data)
    
    # Generazione del range di x per la KDE
    x = np.linspace(filtered_data.min(), filtered_data.max(), 500)
    y = kde(x)
    
    return {
        "mean": filtered_data.mean(),
        "std": filtered_data.std(),
        "x": x,
        "y": y
    }

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
    # plot.show()()

def plot_distribution(angles1, angles2, peaks_or_valleys1, peaks_or_valleys2, title, label1, label2):
    """
    Plotta l'istogramma dei dati e sovrappone la distribuzione normale. In legebnda media e deviazione standard come voci separate.
    """
    # Calcolo delle distribuzioni
    dist1 = calculate_distribution(angles1, peaks_or_valleys1)
    dist2 = calculate_distribution(angles2, peaks_or_valleys2)

    # Stampa statistiche
    print(f"{title}:")
    print(f"  {label1}: mean = {dist1['mean']:.2f}, std = {dist1['std']:.2f}")
    print(f"  {label2}: mean = {dist2['mean']:.2f}, std = {dist2['std']:.2f}")

    plt.figure(figsize=(10, 6))
    plt.hist(angles1[peaks_or_valleys1], bins=60, alpha=0.6, color="red", label=label1, density=True)
    plt.hist(angles2[peaks_or_valleys2], bins=60, alpha=0.6, color="blue", label=label2, density=True)

    # Overlay distribuzioni normali
    plt.plot(dist1["x"], dist1["y"], color="darkred", linestyle="--", label=f"{label1} Fit")
    plt.plot(dist2["x"], dist2["y"], color="darkblue", linestyle="--", label=f"{label2} Fit")

    # Aggiunta di media e deviazione standard come voci separate
    plt.plot([], [], ' ', label=f"{label1}: mean = {dist1['mean']:.2f}, std = {dist1['std']:.2f}")
    plt.plot([], [], ' ', label=f"{label2}: mean = {dist2['mean']:.2f}, std = {dist2['std']:.2f}")

    plt.title(title)
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    # plot.show()()

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
    # plot.show()()

    # Polar plot for minima (valleys)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.scatter(valleys1_angles, valleys1_frames, color="red", label="Setting 1 Valleys")
    ax.scatter(valleys2_angles, valleys2_frames, color="blue", label="Setting 2 Valleys")
    plt.title(f"{title} - Min Angles")
    plt.legend()
    # plot.show()()

def plot_distribution_zones(zones1, zones2, labels, title_prefix):
    """
    Plotta le distribuzioni KDE e gli istogrammi per ciascuna zona.
    """
    colors = ['blue', 'orange', 'green']
    xlim = (min([zone.min() for zone in zones1 + zones2]), max([zone.max() for zone in zones1 + zones2]))

    plt.figure(figsize=(18, 12))

    for i, (zone1, zone2, label, color) in enumerate(zip(zones1, zones2, labels, colors)):
        mean1, std1 = zone1.mean(), zone1.std()
        mean2, std2 = zone2.mean(), zone2.std()

        plt.subplot(2, 3, i + 1)
        plt.hist(zone1, bins=60, density=True, alpha=0.5, label=f'{label} - Histogram 1', color=color)
        kde1 = gaussian_kde(zone1)
        x = np.linspace(zone1.min(), zone1.max(), 500)
        plt.plot(x, kde1(x), color=color, linestyle='--', label=f'{label} - KDE 1')

        plt.title(f"{title_prefix} {label} - Setting 1\nMean: {mean1:.2f}, Std: {std1:.2f}")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Density")
        plt.grid(True)
        plt.xlim(xlim)

        plt.subplot(2, 3, i + 4)
        plt.hist(zone2, bins=60, density=True, alpha=0.5, label=f'{label} - Histogram 2', color=color)
        kde2 = gaussian_kde(zone2)
        x = np.linspace(zone2.min(), zone2.max(), 500)
        plt.plot(x, kde2(x), color=color, linestyle='--', label=f'{label} - KDE 2')

        plt.title(f"{title_prefix} {label} - Setting 2\nMean: {mean2:.2f}, Std: {std2:.2f}")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Density")
        plt.grid(True)
        plt.xlim(xlim)

    plt.tight_layout()
    # plot.show()()

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
output_folder = f"output/LEGS/{ATHLETE_MOD}/plots_output"
os.makedirs(output_folder, exist_ok=True)
output_athlete = ATHLETE_MOD_UC

stats1, peaks1, valleys1, angles1, cycle_amplitudes1, cycle_durations1 = analyze_knee_angle(file_1, angle="knee_l")
stats2, peaks2, valleys2, angles2, cycle_amplitudes2, cycle_durations2 = analyze_knee_angle(file_2, angle="knee_l")

for angle in angles:
    # Analizza l'angolo per entrambi i file
    stats1, peaks1, valleys1, angles1, cycle_amplitudes1, cycle_durations1 = analyze_knee_angle(file_1, angle=angle)
    stats2, peaks2, valleys2, angles2, cycle_amplitudes2, cycle_durations2 = analyze_knee_angle(file_2, angle=angle)

    # # Genera i plot per massimi
    # plot_title = f"Angle Distribution of Maxima (Peaks) - {angle.upper()}"
    # plt.figure(figsize=(10, 6))
    # plot_distribution(
    #     angles1, angles2, 
    #     peaks1, peaks2, 
    #     title=plot_title,
    #     label1="Before Peaks",
    #     label2="After Peaks"
    # )
    # plt.savefig(os.path.join(output_folder, f"{output_athlete}_{angle}_peaks_distribution.png"))
    # plt.close()

    # # Genera i plot per minimi
    # plot_title = f"Angle Distribution of Minima (Valleys) - {angle.upper()}"
    # plt.figure(figsize=(10, 6))
    # plot_distribution(
    #     angles1, angles2, 
    #     valleys1, valleys2, 
    #     title=plot_title,
    #     label1="Before Valleys",
    #     label2="After Valleys"
    # )
    # plt.savefig(os.path.join(output_folder, f"{output_athlete}_{angle}_valleys_distribution.png"))
    # plt.close()

    # # Plot angolo e velocità angolare per un ciclo specifico
    # cycle_index = 0  # Usa il primo ciclo per esempio
    # plt.figure(figsize=(12, 6))
    # plot_angle_and_velocity_for_cycle(angles1, angles2, peaks1, peaks2, cycle_index=cycle_index)
    # plt.savefig(os.path.join(output_folder, f"{output_athlete}_{angle}_cycle_{cycle_index}_angles_velocity.png"))
    # plt.close()
    
print(f"Tutti i plot sono stati salvati nella cartella: {output_folder}")

#___________________VECCHIO CODICE CHE PUò ESSERE UTILE ____________#
# # Print results
# for angle, metrics in comparison_output.items():
#     print(f"{angle}:")
#     for metric, values in metrics.items():
#         print(f"  {metric}: {values}")
#     print()

# #PLOTS
# stats1, peaks1, valleys1, angles1, cycle_amplitudes1, cycle_durations1 = analyze_knee_angle(file_1, angle="knee_l")
# stats2, peaks2, valleys2, angles2, cycle_amplitudes2, cycle_durations2 = analyze_knee_angle(file_2, angle="knee_l")

# plot_angle_and_velocity_for_cycle(angles1, angles2, peaks1, peaks2, cycle_index=75)

# # Generate scatter plot for angle distribution
# # Plot per i massimi (peaks)
# plot_distribution(
#     angles1, angles2, 
#     peaks1, peaks2, 
#     title="Angle Distribution of Maxima (Peaks)",
#     label1="Setting 1 Peaks",
#     label2="Setting 2 Peaks"
# )

# # Plot per i minimi (valleys)
# plot_distribution(
#     angles1, angles2, 
#     valleys1, valleys2, 
#     title="Angle Distribution of Minima (Valleys)",
#     label1="Setting 1 Valleys",
#     label2="Setting 2 Valleys"
# )

# plot_polar_angles_with_frames(peaks1, valleys1, peaks2, valleys2, angles1, angles2, "Angles in time")