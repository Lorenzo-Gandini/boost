import os
import numpy as np
import matplotlib.pyplot as plt
from optitrack.csv_reader import Take
from scipy.stats import gaussian_kde
from utils import get_bones_position, calculate_angles, save_stats, ask_joint_side, user_message
from utils import plot_distribution, plot_polar_angles, plot_angle_and_velocity_for_cycle, analyze_joint_angle 

def run_ankle_analysis(athlete, show_plots):
    """
    Entry point for the ankle analysis from main.
    """
    side = ask_joint_side("ankle")

    if side in ["left", "both"]:
        analyze_single_ankle(athlete, "left", show_plots)
    if side in ["right", "both"]:
        analyze_single_ankle(athlete, "right", show_plots)


def analyze_single_ankle(athlete, side, show_plots):
    """
    Entry point for the single ankle analysis.
    """
    angle_key = "ankle_l" if side == "left" else "ankle_r"

    folder_path = f"lab_records/{athlete}"
    files = sorted([f for f in os.listdir(folder_path) if f.startswith(athlete) and f.endswith(".csv")])

    if len(files) < 2:
        print(f"❌ Impossible to run the analysis for {athlete}. Missing data files.")
        return
    
    csv_file_1 = os.path.join(folder_path, files[0])
    csv_file_2 = os.path.join(folder_path, files[1])

    take_1 = Take().readCSV(csv_file_1)  # Before
    take_2 = Take().readCSV(csv_file_2)  # After

    body_edges_1, bones_pos_1, colors_1 = get_bones_position(take_1)
    angles_1 = calculate_angles(bones_pos_1)

    body_edges_2, bones_pos_2, colors_2 = get_bones_position(take_2)
    angles_2 = calculate_angles(bones_pos_2)

    stats1, peaks1, valleys1, angle_data1 = analyze_joint_angle(angles_1, angle_key)
    stats2, peaks2, valleys2, angle_data2 = analyze_joint_angle(angles_2, angle_key)

    output_folder = f"output/{athlete}/plots/"
    os.makedirs(output_folder, exist_ok=True)

    # Graph for peaks
    plot_distribution(
        angle_data1, angle_data2, peaks1, peaks2,
        f"{side.capitalize()} Ankle Angle Distribution - Peaks",
        "Setting 1 Peaks", "Setting 2 Peaks",
        os.path.join(output_folder, f"{athlete}_ankle_{side}_peaks_distribution.png"),
        show_plots,
    )

    # Graph for valleys
    plot_distribution(
        angle_data1, angle_data2, valleys1, valleys2,
        f"{side.capitalize()} Ankle Angle Distribution - Valleys",
        "Setting 1 Valleys", "Setting 2 Valleys",
        os.path.join(output_folder, f"{athlete}_ankle_{side}_valleys_distribution.png"),
        show_plots,
    )

    # Polar graph for angles peaks
    plot_polar_angles(
        peaks1, peaks2, angle_data1, angle_data2,
        f"{side.capitalize()} Ankle Polar Plot - Peaks",
        "Setting 1 Peaks", "Setting 2 Peaks",
        os.path.join(output_folder, f"{athlete}_ankle_{side}_polar_peaks.png"),
        show_plots,
    )

    # Polar graph for angles valleys
    plot_polar_angles(
        valleys1, valleys2, angle_data1, angle_data2,
        f"{side.capitalize()} Ankle Polar Plot - Valleys",
        "Setting 1 Valleys", "Setting 2 Valleys",
        os.path.join(output_folder, f"{athlete}_ankle_{side}_polar_valleys.png"),
        show_plots,
    )

    cycle_index = 0  # Define the cycle you want to analyze and plot the angle and velocity
    plot_angle_and_velocity_for_cycle(
        angle_data1, angle_data2, peaks1, peaks2, cycle_index,
        f"{side.capitalize()} Ankle - Single Cycle Analysis",
        "Setting 1", "Setting 2",
        os.path.join(output_folder, f"{athlete}_ankle_{side}_cycle_{cycle_index}_analysis.png"),
        show_plots,
    )

    # Save statistics
    user_message(f"All graphs for the {side} ankle analysis have been saved in {output_folder}.", "saving_graph")
    save_stats(stats1, stats2, athlete, "ankle", side)
