import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from utils import get_bones_position, calculate_angles, save_stats, ask_joint_side, user_message
from utils import plot_distribution, plot_polar_angles, plot_angle_and_velocity_for_cycle, analyze_joint_angle   
from optitrack.csv_reader import Take

def run_knee_analysis(athlete, athlete_mod, athlete_mod_uc, show_plots):
    """
    Entry point for knee analysis.
    """
    side = ask_joint_side("knee")

    if side in ["left", "both"]:
        analyze_single_knee(athlete, athlete_mod, athlete_mod_uc, "left", show_plots)
    if side in ["right", "both"]:
        analyze_single_knee(athlete, athlete_mod, athlete_mod_uc, "right", show_plots)

def analyze_single_knee(athlete, athlete_mod, athlete_mod_uc, side, show_plots):
    """
    Entry point for the single knee analysis.
    """
    angle_key = "knee_l" if side == "left" else "knee_r"

    csv_file_1 = f"lab_records/{athlete_mod}_1.csv"
    csv_file_2 = f"lab_records/{athlete_mod}_2.csv"

    if not (os.path.exists(csv_file_1) and os.path.exists(csv_file_2)):
        print(f"Impossible to run the analysis for {athlete}. Missing data files.")
        return

    take_1 = Take().readCSV(csv_file_1) # Before
    take_2 = Take().readCSV(csv_file_2) # After

    body_edges_1, bones_pos_1, colors_1 = get_bones_position(take_1)
    angles_1 = calculate_angles(bones_pos_1)

    body_edges_2, bones_pos_2, colors_2 = get_bones_position(take_2)
    angles_2 = calculate_angles(bones_pos_2)

    stats1, peaks1, valleys1, angle_data1 = analyze_joint_angle(angles_1, angle_key)
    stats2, peaks2, valleys2, angle_data2 = analyze_joint_angle(angles_2, angle_key)

    output_folder = f"output/{athlete}/"
    os.makedirs(output_folder, exist_ok=True)

    # Peaks distribution
    plot_distribution(
        angle_data1, angle_data2, peaks1, peaks2,
        f"{side.capitalize()} Knee Angle Distribution - Peaks",
        "Setting 1 Peaks", "Setting 2 Peaks",
        os.path.join(output_folder, f"plots/{athlete_mod_uc}_knee_{side}_peaks_distribution.png"),
        show_plots,
    )

    # Valleys distribution
    plot_distribution(
        angle_data1, angle_data2, valleys1, valleys2,
        f"{side.capitalize()} Knee Angle Distribution - Valleys",
        "Setting 1 Valleys", "Setting 2 Valleys",
        os.path.join(output_folder, f"plots/{athlete_mod_uc}_knee_{side}_valleys_distribution.png"),
        show_plots,
    )

    # Polar graph - peaks
    plot_polar_angles(
        peaks1, peaks2, angle_data1, angle_data2,
        f"{side.capitalize()} Knee Polar Plot - Peaks",
        "Setting 1 Peaks", "Setting 2 Peaks",
        os.path.join(output_folder, f"plots/{athlete_mod_uc}_knee_{side}_polar_peaks.png"),
        show_plots,
    )

    # Polar graph - valleys
    plot_polar_angles(
        valleys1, valleys2, angle_data1, angle_data2,
        f"{side.capitalize()} Knee Polar Plot - Valleys",
        "Setting 1 Valleys", "Setting 2 Valleys",
        os.path.join(output_folder, f"plots/{athlete_mod_uc}_knee_{side}_polar_valleys.png"),
        show_plots,
    )

    cycle_index = 75  # Choose the cycle you want to analyze
    plot_angle_and_velocity_for_cycle(
        angle_data1, angle_data2, peaks1, peaks2, cycle_index,
        f"{side.capitalize()} Knee - Single Cycle Analysis",
        "Setting 1", "Setting 2",
        os.path.join(output_folder, f"plots/{athlete_mod_uc}_knee_{side}_cycle_{cycle_index}_analysis.png"),
        show_plots,
    )

    # Save statistics
    user_message(f"All graphs for the {side} knee analysis have been saved.", "saving_graph")
    save_stats(stats1, stats2, athlete, athlete_mod_uc, "knee", side)
