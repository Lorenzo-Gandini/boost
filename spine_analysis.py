import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from optitrack.csv_reader import Take
from utils import get_bones_position, analyze_angle_from_points, plot_angles_combined, plot_oscillations, get_zones, calculate_oscillations, save_stats, user_message

def run_spine_analysis(athlete, show_plots):
    """
    Entry point for spine analysis, including stomach angles, spine-ground angles, and oscillations.
    """
    folder_path = f"lab_records/{athlete}"
    files = sorted([f for f in os.listdir(folder_path) if f.startswith(athlete) and f.endswith(".csv")])

    if len(files) < 2:
        print(f"❌ Impossible to run the analysis for {athlete}. Missing data files.")
        return

    csv_file_1 = os.path.join(folder_path, files[0])
    csv_file_2 = os.path.join(folder_path, files[1])

    take_1 = Take().readCSV(csv_file_1)
    take_2 = Take().readCSV(csv_file_2)

    body_edges_1, bones_pos_1, _ = get_bones_position(take_1)
    body_edges_2, bones_pos_2, _ = get_bones_position(take_2)

    output_folder = f"output/{athlete}/plots/"
    os.makedirs(output_folder, exist_ok=True)

    # Split data into training zones
    zones_setting_1 = get_zones(pd.DataFrame(bones_pos_1.reshape(bones_pos_1.shape[0], -1)))
    zones_setting_2 = get_zones(pd.DataFrame(bones_pos_2.reshape(bones_pos_2.shape[0], -1)))

    # --- Stomach Angle Analysis ---
    stomach_angles_1 = [analyze_angle_from_points(zone.values.reshape(-1, 21, 3), 0, 1, 1, 2) for zone in zones_setting_1] 
    stomach_angles_2 = [analyze_angle_from_points(zone.values.reshape(-1, 21, 3), 0, 1, 1, 2) for zone in zones_setting_2] 

    stomach_angles_1_series = [item[0] for item in stomach_angles_1]
    stomach_angles_2_series = [item[0] for item in stomach_angles_2]

    plot_angles_combined(
        stomach_angles_1_series, stomach_angles_2_series,
        ["Zone 2", "Zone 3", "Zone 5"], "Stomach Angle Analysis",
        os.path.join(output_folder, f"{athlete}_stomach_angle.png"),
        show_plots
    )

    # --- Spine-Ground Angle Analysis ---
    spine_angles_1 = [analyze_angle_from_points(zone.values.reshape(-1, 21, 3), 0, 2, 'ground', 'ground') for zone in zones_setting_1] 
    spine_angles_2 = [analyze_angle_from_points(zone.values.reshape(-1, 21, 3), 0, 2, 'ground', 'ground') for zone in zones_setting_2] 

    spine_angles_1_series = [item[0] for item in spine_angles_1]
    spine_angles_2_series = [item[0] for item in spine_angles_2]

    plot_angles_combined(
        spine_angles_1_series, spine_angles_2_series,
        ["Zone 2", "Zone 3", "Zone 5"], "Spine Ground Angle Analysis",
        os.path.join(output_folder, f"{athlete}_spine_ground_angle.png"),
        show_plots
    )

    # Oscillations and statistics
    all_oscillations_1, oscillations_stats_1 = calculate_oscillations(bones_pos_1, zones_setting_1)
    all_oscillations_2, oscillations_stats_2 = calculate_oscillations(bones_pos_2, zones_setting_2)

    plot_oscillations(
        all_oscillations_1, all_oscillations_2,
        "Combined Oscillations Analysis",
        os.path.join(output_folder, f"{athlete}_oscillations_analysis_combined.png"), 
        show_plots
    )

    user_message(f"All graphs for the spine analysis have been saved in {output_folder}.", "saving_graph")
    save_stats(oscillations_stats_1, oscillations_stats_2, athlete, "spine")
