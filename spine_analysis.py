import os
import json
import pandas as pd
from optitrack.csv_reader import Take
from utils import (
    get_bones_position,
    analyze_angle_from_points,
    plot_angles_combined,
    plot_oscillations,
    get_zones,
    calculate_oscillations
)

def run_spine_analysis(athlete, athlete_mod, athlete_mod_uc, show_plots):
    """
    Entry point for spine analysis, including stomach angles, spine-ground angles, and oscillations.
    """
    csv_file_1 = f"lab_records/{athlete_mod}_1.csv"
    csv_file_2 = f"lab_records/{athlete_mod}_2.csv"

    # Check for data files
    if not (os.path.exists(csv_file_1) and os.path.exists(csv_file_2)):
        print(f"Impossible to run the analysis for {athlete}. Missing data files.")
        return

    # Load data from CSV files
    take_1 = Take().readCSV(csv_file_1)
    take_2 = Take().readCSV(csv_file_2)

    body_edges_1, bones_pos_1, _ = get_bones_position(take_1)
    body_edges_2, bones_pos_2, _ = get_bones_position(take_2)

    output_folder_plot = f"output/{athlete}/plots/"
    output_folder_stats = f"output/{athlete}/stats/"
    os.makedirs(output_folder_plot, exist_ok=True)
    os.makedirs(output_folder_stats, exist_ok=True)


    # Split data into zones
    zones_setting_1 = get_zones(pd.DataFrame(bones_pos_1.reshape(bones_pos_1.shape[0], -1)))
    zones_setting_2 = get_zones(pd.DataFrame(bones_pos_2.reshape(bones_pos_2.shape[0], -1)))

    # --- Stomach Angle Analysis ---
    stomach_angles_1 = [analyze_angle_from_points(zone.values.reshape(-1, 21, 3), 0, 1, 1, 2) for zone in zones_setting_1]  # Restituisce tuple
    stomach_angles_2 = [analyze_angle_from_points(zone.values.reshape(-1, 21, 3), 0, 1, 1, 2) for zone in zones_setting_2]  # Restituisce tuple

    # Estrai solo le serie per il plotting
    stomach_angles_1_series = [item[0] for item in stomach_angles_1]
    stomach_angles_2_series = [item[0] for item in stomach_angles_2]

    plot_angles_combined(
        stomach_angles_1_series, stomach_angles_2_series,
        ["Zone 2", "Zone 3", "Zone 5"], "Stomach Angle Analysis",
        os.path.join(output_folder_plot, f"{athlete_mod_uc}_stomach_angle.png"),
        show_plots
    )


    # --- Spine-Ground Angle Analysis ---
    spine_angles_1 = [analyze_angle_from_points(zone.values.reshape(-1, 21, 3), 0, 2, 'ground', 'ground') for zone in zones_setting_1]  # Restituisce tuple
    spine_angles_2 = [analyze_angle_from_points(zone.values.reshape(-1, 21, 3), 0, 2, 'ground', 'ground') for zone in zones_setting_2]  # Restituisce tuple

    # Estrai solo le serie per il plotting
    spine_angles_1_series = [item[0] for item in spine_angles_1]
    spine_angles_2_series = [item[0] for item in spine_angles_2]

    plot_angles_combined(
        spine_angles_1_series, spine_angles_2_series,
        ["Zone 2", "Zone 3", "Zone 5"], "Spine Ground Angle Analysis",
        os.path.join(output_folder_plot, f"{athlete_mod_uc}_spine_ground_angle.png"),
        show_plots
    )

    # Calcola le oscillazioni e le statistiche
    all_oscillations_1, oscillations_stats_1 = calculate_oscillations(bones_pos_1, zones_setting_1)
    all_oscillations_2, oscillations_stats_2 = calculate_oscillations(bones_pos_2, zones_setting_2)

    # Plot delle oscillazioni
    plot_oscillations(
        all_oscillations_1, all_oscillations_2,
        "Combined Oscillations Analysis",
        os.path.join(output_folder_plot, f"{athlete_mod_uc}_oscillations_analysis_combined.png"), 
        show_plots
    )

    oscillations_stats = {
        "Setting_1": {
            "global": oscillations_stats_1["global"],
            "zones": oscillations_stats_1["zones"]
        },
        "Setting_2": {
            "global": oscillations_stats_2["global"],
            "zones": oscillations_stats_2["zones"]
        }
    }

    # Definizione dei nomi delle zone
    zone_names = ["Zone 2", "Zone 3", "Zone 5"]

    # Salva le statistiche in formato JSON
    # --- Save all statistics ---

    results = {
        "stomach_angles": {
            "Setting_1": [{"zone": zone_names[i], **stomach_angles_1[i][1]} for i in range(len(stomach_angles_1))],
            "Setting_2": [{"zone": zone_names[i], **stomach_angles_2[i][1]} for i in range(len(stomach_angles_2))]
        },
        "spine_angles": {
            "Setting_1": [{"zone": zone_names[i], **spine_angles_1[i][1]} for i in range(len(spine_angles_1))],
            "Setting_2": [{"zone": zone_names[i], **spine_angles_2[i][1]} for i in range(len(spine_angles_2))]
        },
        "oscillations": oscillations_stats,
    }


    output_file = os.path.join(output_folder_stats, f"{athlete_mod_uc}_spine_analysis_stats.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)