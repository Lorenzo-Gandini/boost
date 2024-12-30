import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from utils import get_bones_position, calculate_angles, save_stats
from optitrack.csv_reader import Take


def ask_knee_side():
    """
    Ask to the user which knee wants to analyze (Right or Left).
    """
    valid_inputs = {
            "r": "right", "right": "right", "1": "right",
            "l": "left", "left": "left", "2": "left",
            "b": "both", "both": "both", "3": "both",
        }
    while True:
        side = input("Which knee do you want to analyze? OPTIONS:\n"
                    "1. Right knee\n"
                    "2. Left knee\n"
                    "3. Both knees\n").strip().lower()
        if side in valid_inputs:
            return valid_inputs[side]
        print("Invalid input. Please type 'left', 'right', or 'both.")


def analyze_knee_angle(angles, angle_key):
    """
    Analyze a specific knee angle from the extracted angles.
    """
    angle_data = np.array([frame[angle_key] for frame in angles])

    # Trova picchi (massimi) e valli (minimi)
    peaks, _ = find_peaks(angle_data, prominence=5)
    valleys, _ = find_peaks(-angle_data, prominence=5)

    # Metriche dei cicli
    num_cycles = min(len(peaks), len(valleys))
    peaks = peaks[:num_cycles]
    valleys = valleys[:num_cycles]

    cycle_amplitudes = angle_data[peaks] - angle_data[valleys]
    cycle_durations = np.diff(peaks)
    cadence = 60 / (np.mean(cycle_durations) / 30) if len(cycle_durations) > 0 else 0  # 30 FPS assumed

    # Velocità angolare
    angular_velocity = np.gradient(angle_data, 1 / 30)  # 30 FPS assumed
    angular_velocity_mean = np.mean(angular_velocity)
    angular_velocity_std = np.std(angular_velocity)

    stats = {
        "mean": np.mean(angle_data),
        "std": np.std(angle_data),
        "min": np.min(angle_data),
        "max": np.max(angle_data),
        "range": np.max(angle_data) - np.min(angle_data),
        "cycle_amplitude_mean": np.mean(cycle_amplitudes) if len(cycle_amplitudes) > 0 else 0,
        "cycle_amplitude_std": np.std(cycle_amplitudes) if len(cycle_amplitudes) > 0 else 0,
        "cadence": cadence,
        "angular_velocity_mean": angular_velocity_mean,
        "angular_velocity_std": angular_velocity_std,
    }

    return stats, peaks, valleys, angle_data


def plot_distribution(angles1, angles2, indices1, indices2, title, label1, label2, output_file, show_plots):
    """
    Plot the histogram and KDE of angles for peaks or valleys.
    Adjusts labels and titles dynamically based on the data passed.
    """
    def calculate_distribution(data, indices):
        filtered_data = data[indices]
        kde = gaussian_kde(filtered_data)
        x = np.linspace(filtered_data.min(), filtered_data.max(), 500)
        y = kde(x)
        return {"mean": filtered_data.mean(), "std": filtered_data.std(), "x": x, "y": y}

    dist1 = calculate_distribution(angles1, indices1)
    dist2 = calculate_distribution(angles2, indices2)

    print(f"Saving distribution plot to: {output_file}")
    plt.figure(figsize=(10, 6))
    plt.hist(angles1[indices1], bins=60, alpha=0.6, color="red", label=label1, density=True)
    plt.hist(angles2[indices2], bins=60, alpha=0.6, color="blue", label=label2, density=True)
    plt.plot(dist1["x"], dist1["y"], color="darkred", linestyle="--", label=f"{label1} Fit")
    plt.plot(dist2["x"], dist2["y"], color="darkblue", linestyle="--", label=f"{label2} Fit")
    plt.title(title)
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()

    # Save and optionally show the plot
    plt.savefig(output_file)
    print(f"Distribution plot saved successfully: {output_file}")
    if show_plots:
        plt.show()
    plt.close()

def plot_polar_angles(indices1, indices2, angles1, angles2, title, label1, label2, output_file, show_plots):
    """
    Plot angles in polar coordinates using frame indices as the radial coordinate.
    Supports both peaks and valleys by adjusting labels and titles.
    """
    # Convert angles to radians
    angles1_rad = np.deg2rad(angles1[indices1])
    angles2_rad = np.deg2rad(angles2[indices2])

    print(f"Saving polar plot to: {output_file}")
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Plot the data
    ax.scatter(angles1_rad, indices1, color="red", label=label1)
    ax.scatter(angles2_rad, indices2, color="blue", label=label2)

    plt.title(title)
    plt.legend()

    # Save and optionally show the plot
    plt.savefig(output_file)
    print(f"Polar plot saved successfully: {output_file}")
    if show_plots:
        plt.show()
    plt.close()

def plot_angle_and_velocity_for_cycle( angles1, angles2, peaks1, peaks2, cycle_index, title, label1, label2, output_file, show_plots, frame_rate=30):
    """
    Plot angle and angular velocity for a specific cycle for two settings.
    """
    # Estrai i dati del ciclo per il Setting 1
    start_frame1 = peaks1[cycle_index]
    end_frame1 = peaks1[cycle_index + 1]
    angles_cycle1 = angles1[start_frame1:end_frame1]
    velocity_cycle1 = np.gradient(angles_cycle1, 1 / frame_rate)

    # Estrai i dati del ciclo per il Setting 2
    start_frame2 = peaks2[cycle_index]
    end_frame2 = peaks2[cycle_index + 1]
    angles_cycle2 = angles2[start_frame2:end_frame2]
    velocity_cycle2 = np.gradient(angles_cycle2, 1 / frame_rate)

    # Crea il grafico
    print(f"Saving single cycle analysis plot to: {output_file}")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Angoli sul primo asse Y
    ax1.plot(angles_cycle1, label=f"{label1} - Angle", color="red", linestyle="-")
    ax1.plot(angles_cycle2, label=f"{label2} - Angle", color="blue", linestyle="-")
    ax1.set_xlabel("Frame (within cycle)")
    ax1.set_ylabel("Angle (degrees)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left")
    ax1.grid()

    # Velocità angolare sul secondo asse Y
    ax2 = ax1.twinx()
    ax2.plot(velocity_cycle1, label=f"{label1} - Angular Velocity", color="red", linestyle="--")
    ax2.plot(velocity_cycle2, label=f"{label2} - Angular Velocity", color="blue", linestyle="--")
    ax2.set_ylabel("Angular Velocity (degrees/second)", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.legend(loc="upper right")

    # Salva il grafico
    plt.title(title)
    plt.savefig(output_file)
    print(f"Single cycle plot saved successfully: {output_file}")
    if show_plots:
        plt.show()
    plt.close()

def run_knee_analysis(athlete, athlete_mod, athlete_mod_uc, show_plots):
    """
    Entry point for knee analysis.
    """
    side = ask_knee_side()

    if side in ["left", "both"]:
        analyze_single_knee(athlete, athlete_mod, athlete_mod_uc, "left", show_plots)
    if side in ["right", "both"]:
        analyze_single_knee(athlete, athlete_mod, athlete_mod_uc, "right", show_plots)

def analyze_single_knee(athlete, athlete_mod, athlete_mod_uc, side, show_plots):
    angle_key = "knee_l" if side == "left" else "knee_r"

    csv_file_1 = f"lab_records/{athlete_mod}_1.csv"
    csv_file_2 = f"lab_records/{athlete_mod}_2.csv"

    if not (os.path.exists(csv_file_1) and os.path.exists(csv_file_2)):
        print(f"Impossible to run the analysis for {athlete}. Missing data files.")
        return


    take_1 = Take().readCSV(csv_file_1)
    take_2 = Take().readCSV(csv_file_2)

    body_edges_1, bones_pos_1, colors_1 = get_bones_position(take_1)
    angles_1 = calculate_angles(bones_pos_1)

    body_edges_2, bones_pos_2, colors_2 = get_bones_position(take_2)
    angles_2 = calculate_angles(bones_pos_2)

    stats1, peaks1, valleys1, angle_data1 = analyze_knee_angle(angles_1, angle_key)
    stats2, peaks2, valleys2, angle_data2 = analyze_knee_angle(angles_2, angle_key)

    output_folder = f"output/{athlete}/"
    os.makedirs(output_folder, exist_ok=True)

    # Distribuzione degli angoli - picchi
    plot_distribution(
        angle_data1, angle_data2, peaks1, peaks2,
        f"{side.capitalize()} Knee Angle Distribution - Peaks",
        "Setting 1 Peaks", "Setting 2 Peaks",
        os.path.join(output_folder, f"plots/{athlete_mod_uc}_knee_{side}_peaks_distribution.png"),
        show_plots,
    )

    # Distribuzione per i minimi
    plot_distribution(
        angle_data1, angle_data2, valleys1, valleys2,
        f"{side.capitalize()} Knee Angle Distribution - Valleys",
        "Setting 1 Valleys", "Setting 2 Valleys",
        os.path.join(output_folder, f"plots/{athlete_mod_uc}_knee_{side}_valleys_distribution.png"),
        show_plots,
    )

    # Grafico polare per i picchi
    plot_polar_angles(
        peaks1, peaks2, angle_data1, angle_data2,
        f"{side.capitalize()} Knee Polar Plot - Peaks",
        "Setting 1 Peaks", "Setting 2 Peaks",
        os.path.join(output_folder, f"plots/{athlete_mod_uc}_knee_{side}_polar_peaks.png"),
        show_plots,
    )

    # Grafico polare per i minimi
    plot_polar_angles(
        valleys1, valleys2, angle_data1, angle_data2,
        f"{side.capitalize()} Knee Polar Plot - Valleys",
        "Setting 1 Valleys", "Setting 2 Valleys",
        os.path.join(output_folder, f"plots/{athlete_mod_uc}_knee_{side}_polar_valleys.png"),
        show_plots,
    )

    cycle_index = 75  # Usa il primo ciclo per esempio
    plot_angle_and_velocity_for_cycle(
        angle_data1, angle_data2, peaks1, peaks2, cycle_index,
        f"{side.capitalize()} Knee - Single Cycle Analysis",
        "Setting 1", "Setting 2",
        os.path.join(output_folder, f"plots/{athlete_mod_uc}_knee_{side}_cycle_{cycle_index}_analysis.png"),
        show_plots,
    )

        # Save statistics
    save_stats(stats1, stats2, athlete, athlete_mod_uc, "knee", side)
