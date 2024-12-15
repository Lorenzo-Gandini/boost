'''
Back Biomechanical Analysis
This file analyzes spine oscillations and the angle between the spine and the ground, between two different settings.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_spine_data, calculate_angle, get_zones
from scipy.stats import gaussian_kde

#--- FUNCTIONS ---#

def analyze_stomach_angle(df):
    """
    Analyze stomach angle (between hip-ab and ab-chest).
    """
    return analyze_angle(df, 'hip', 'ab', 'ab', 'chest', "stomach_angle")

def analyze_angle(df, point1_start, point1_end, point2_start, point2_end, angle_name):
    """
    Extract the angle between two given vectors
    """
    angles = []

    for _, row in df.iterrows():
        start1 = np.array([row[f'{point1_start}_x'], row[f'{point1_start}_y'], row[f'{point1_start}_z']])
        end1 = np.array([row[f'{point1_end}_x'], row[f'{point1_end}_y'], row[f'{point1_end}_z']])
        start2 = np.array([row[f'{point2_start}_x'], row[f'{point2_start}_y'], row[f'{point2_start}_z']])
        end2 = np.array([row[f'{point2_end}_x'], row[f'{point2_end}_y'], row[f'{point2_end}_z']])

        vec1 = end1 - start1
        vec2 = end2 - start2

        if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
            angle = 180 - calculate_angle(vec1, vec2)  
            angles.append(angle)
        else:
            angles.append(None)  

    angles = pd.Series(angles, name=angle_name)

    stats = {
        'mean_angle': angles.mean(),
        'std_angle': angles.std(),
        'max_angle': angles.max(),
        'min_angle': angles.min()
    }

    return angles, stats

def analyze_hip_chest_ground_angle(df):
    """
    Extract angle between spine and ground
    """
    ground_vector = np.array([0, 1, 0])  # Vettore orizzontale nel piano terreno
    angles = []

    for _, row in df.iterrows():
        hip = np.array([row['hip_x'], row['hip_y'], row['hip_z']])
        chest = np.array([row['chest_x'], row['chest_y'], row['chest_z']])

        hip_chest = chest - hip

        if np.linalg.norm(hip_chest) > 0:
            angle = calculate_angle(hip_chest, ground_vector)
            angles.append(angle)
        else:
            angles.append(None)  # Gestisce frame invalidi

    angles = pd.Series(angles, name="hip_chest_ground_angle")

    stats = {
        'mean_angle': angles.mean(),
        'std_angle': angles.std(),
        'max_angle': angles.max(),
        'min_angle': angles.min()
    }

    return angles, stats

def plot_angles_combined(zone2_1, zone3_1, zone5_1, zone2_2, zone3_2, zone5_2, title_prefix):
    """
    2x3 plot with histograms and KDE ìfor both settings and all zones of training.
    """
    zones_1 = [zone2_1.dropna(), zone3_1.dropna(), zone5_1.dropna()]
    zones_2 = [zone2_2.dropna(), zone3_2.dropna(), zone5_2.dropna()]
    labels = ['Zona 2', 'Zona 3', 'Zona 5']
    colors = ['blue', 'orange', 'green']

    all_data = pd.concat(zones_1 + zones_2)
    xlim = (all_data.min(), all_data.max())

    plt.figure(figsize=(18, 12))

    for i, (zone, label, color) in enumerate(zip(zones_1, labels, colors), 1):
        mean_angle = zone.mean()
        std_angle = zone.std()

        plt.subplot(2, 3, i)
        plt.hist(zone, bins=30, density=True, alpha=0.5, label=f'{label} - Histogram', color=color)

        kde = gaussian_kde(zone)
        x = np.linspace(zone.min(), zone.max(), 500)
        plt.plot(x, kde(x), color=color, linestyle='--', label=f'{label} - KDE')

        plt.title(f"{title_prefix} {label} - Setting 1\nMean: {mean_angle:.2f}, Std: {std_angle:.2f}")
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Density')
        plt.grid(True)
        plt.ylim(0, 1.5)  # Scala uniforme per tutti i grafici
        plt.xlim(*xlim)  # Scala uniforme sull'asse X

    for i, (zone, label, color) in enumerate(zip(zones_2, labels, colors), 4):
        mean_angle = zone.mean()
        std_angle = zone.std()

        plt.subplot(2, 3, i)
        plt.hist(zone, bins=30, density=True, alpha=0.5, label=f'{label} - Histogram', color=color)

        kde = gaussian_kde(zone)
        x = np.linspace(zone.min(), zone.max(), 500)
        plt.plot(x, kde(x), color=color, linestyle='--', label=f'{label} - KDE')

        plt.title(f"{title_prefix} {label} - Setting 2\nMean: {mean_angle:.2f}, Std: {std_angle:.2f}")
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Density')
        plt.grid(True)
        plt.ylim(0, 1.5)  # Scala uniforme per tutti i grafici
        plt.xlim(*xlim)  # Scala uniforme sull'asse X

    plt.tight_layout()
    plt.show()


def calculate_ab_oscillation_to_axis_with_stats(df, zone_name):
    """
    Calcola le oscillazioni laterali del punto 'ab' rispetto all'asse 'hip-chest',
    distinguendo destra (positivo) e sinistra (negativo), e restituisce le statistiche.

    Parameters:
        df (pd.DataFrame): Dati contenenti le coordinate di 'hip', 'chest' e 'ab'.
        zone_name (str): Nome della zona da utilizzare per il titolo.

    Returns:
        oscillations (pd.DataFrame): DataFrame contenente il tempo e le oscillazioni con segno.
        stats (dict): Statistiche per destra e sinistra.
    """
    oscillations = []
    times = []

    for index, row in df.iterrows():
        hip = np.array([row['hip_x'], row['hip_y'], row['hip_z']])
        chest = np.array([row['chest_x'], row['chest_y'], row['chest_z']])
        ab = np.array([row['ab_x'], row['ab_y'], row['ab_z']])

        # Vettori hip-chest e hip-ab
        hip_chest = chest - hip
        hip_ab = ab - hip

        if np.linalg.norm(hip_chest) > 0:
            # Proiezione di hip-ab su hip-chest
            projection = (np.dot(hip_ab, hip_chest) / np.linalg.norm(hip_chest)**2) * hip_chest
            lateral_oscillation = hip_ab - projection  # Componente ortogonale

            # Segno usando il prodotto vettoriale per distinguere destra e sinistra
            cross_product = np.cross(hip_chest, lateral_oscillation)
            sign = np.sign(cross_product[-1])  # Considera la componente Z per il segno

            oscillations.append(sign * np.linalg.norm(lateral_oscillation))
        else:
            oscillations.append(None)  # Gestisce frame invalidi

        times.append(index)  # Salva il frame o il tempo relativo

    oscillations_df = pd.DataFrame({"time": times, "oscillation": oscillations})

    # Calcola le statistiche
    positive_oscillations = oscillations_df['oscillation'][oscillations_df['oscillation'] > 0]
    negative_oscillations = oscillations_df['oscillation'][oscillations_df['oscillation'] < 0]

    stats = {
        'zone': zone_name,
        'right_mean': positive_oscillations.mean(),
        'right_std': positive_oscillations.std(),
        'left_mean': negative_oscillations.mean(),
        'left_std': negative_oscillations.std()
    }

    return oscillations_df, stats

def plot_zone_statistics(stats_list, title):
    """
    Plotta le statistiche (media e deviazione standard) per le oscillazioni a destra e sinistra
    su un unico asse con tre sottografici.

    Parameters:
        stats_list (list of dict): Lista di statistiche per ciascuna zona.
        title (str): Titolo del grafico.
    """
    zones = [s['zone'] for s in stats_list]
    right_means = [s['right_mean'] for s in stats_list]
    right_stds = [s['right_std'] for s in stats_list]
    left_means = [s['left_mean'] for s in stats_list]
    left_stds = [s['left_std'] for s in stats_list]

    fig, axes = plt.subplots(1, len(stats_list), figsize=(15, 5), sharex=True, sharey=True)

    for i, ax in enumerate(axes):
        zone = zones[i]
        ax.barh([0], [right_means[i]], xerr=[right_stds[i]], color='blue', alpha=0.7, capsize=5, label='Right')
        ax.barh([0], [left_means[i]], xerr=[left_stds[i]], color='orange', alpha=0.7, capsize=5, label='Left')
        ax.set_title(zone)
        ax.set_xlabel('Oscillation (mm)')
        ax.axvline(0, color='red', linestyle='--')
        ax.set_yticks([])

    fig.suptitle(title)
    axes[0].legend(loc='upper left')
    plt.tight_layout()
    plt.show()


#---- PROGRAM ---#
# Load files
file_1 = "output/spine_metrics_1.json"
file_2 = "output/spine_metrics_2.json"

df_1 = load_spine_data(file_1)
df_2 = load_spine_data(file_2)

# Estrazione delle zone
zone2_1, zone3_1, zone5_1 = get_zones(df_1)
zone2_2, zone3_2, zone5_2 = get_zones(df_2)

# Stomach angles and plot
stomach_zone2_1, _ = analyze_stomach_angle(zone2_1)
stomach_zone3_1, _ = analyze_stomach_angle(zone3_1)
stomach_zone5_1, _ = analyze_stomach_angle(zone5_1)

stomach_zone2_2, _ = analyze_stomach_angle(zone2_2)
stomach_zone3_2, _ = analyze_stomach_angle(zone3_2)
stomach_zone5_2, _ = analyze_stomach_angle(zone5_2)

# # Plot combinato
plot_angles_combined(stomach_zone2_1, stomach_zone3_1, stomach_zone5_1, stomach_zone2_2, stomach_zone3_2, stomach_zone5_2, "Stomach Angle")

# Spine-ground angle and plot
hip_chest_ground_zone2_1, _ = analyze_hip_chest_ground_angle(zone2_1)
hip_chest_ground_zone3_1, _ = analyze_hip_chest_ground_angle(zone3_1)
hip_chest_ground_zone5_1, _ = analyze_hip_chest_ground_angle(zone5_1)

hip_chest_ground_zone2_2, _ = analyze_hip_chest_ground_angle(zone2_2)
hip_chest_ground_zone3_2, _ = analyze_hip_chest_ground_angle(zone3_2)
hip_chest_ground_zone5_2, _ = analyze_hip_chest_ground_angle(zone5_2)
plot_angles_combined(hip_chest_ground_zone2_1, hip_chest_ground_zone3_1, hip_chest_ground_zone5_1,
                     hip_chest_ground_zone2_2, hip_chest_ground_zone3_2, hip_chest_ground_zone5_2,
                     "Hip-Chest Ground Angle")

# Oscillations and plots
oscillation_data_zone2_1, stats_zone2_1 = calculate_ab_oscillation_to_axis_with_stats(zone2_1, 'Zone 2 - Setting 1')
oscillation_data_zone3_1, stats_zone3_1 = calculate_ab_oscillation_to_axis_with_stats(zone3_1, 'Zone 3 - Setting 1')
oscillation_data_zone5_1, stats_zone5_1 = calculate_ab_oscillation_to_axis_with_stats(zone5_1, 'Zone 5 - Setting 1')

oscillation_data_zone2_2, stats_zone2_2 = calculate_ab_oscillation_to_axis_with_stats(zone2_2, 'Zone 2 - Setting 2')
oscillation_data_zone3_2, stats_zone3_2 = calculate_ab_oscillation_to_axis_with_stats(zone3_2, 'Zone 3 - Setting 2')
oscillation_data_zone5_2, stats_zone5_2 = calculate_ab_oscillation_to_axis_with_stats(zone5_2, 'Zone 5 - Setting 2')

plot_zone_statistics([stats_zone2_1, stats_zone3_1, stats_zone5_1], "Average Oscillation by Zone - Setting 1")
plot_zone_statistics([stats_zone2_2, stats_zone3_2, stats_zone5_2], "Average Oscillation by Zone - Setting 2")




# VECCHIO CODICE CHE PUò ESSERE UTILE

# def calculate_ab_statistics(df):
#     '''
#     Provides stats and the oscillations about the "ab" (=abdomen) point
#     '''
#     ab_x, ab_y, ab_z = df["ab_x"], df["ab_y"], df["ab_z"]

#     oscillations = df["ab_y"] - df["ab_y"].mean()

#     # Calculate differences to estimate velocities
#     velocities = np.sqrt(np.diff(ab_x)**2 + np.diff(ab_y)**2 + np.diff(ab_z)**2)
#     mean_velocity = np.mean(velocities)
#     std_velocity = np.std(velocities)
    
#     stats = {
#         "mean_x": ab_x.mean(),
#         "std_x": ab_x.std(),
#         "range_x": ab_x.max() - ab_x.min(),
#         "mean_y": ab_y.mean(),
#         "std_y": ab_y.std(),
#         "range_y": ab_y.max() - ab_y.min(),
#         "mean_z": ab_z.mean(),
#         "std_z": ab_z.std(),
#         "range_z": ab_z.max() - ab_z.min(),
#         "mean_velocity": mean_velocity,
#         "std_velocity": std_velocity,
#     }
    
#     return pd.Series(oscillations, name="ab_oscillations"), stats

# #SPINE ALIGNMENT TO THE GROUND
# def calculate_spine_alignment(df):
#     """
#     Calculate alignment angles of the spine segments (hip-ab and ab-chest) and relative to the ground.
#     """
#     angles = []

#     for idx, row in df.iterrows():
#         hip = np.array([row["hip_x"], row["hip_y"], row["hip_z"]])
#         ab = np.array([row["ab_x"], row["ab_y"], row["ab_z"]])
#         chest = np.array([row["chest_x"], row["chest_y"], row["chest_z"]])

#         # Vectors
#         hip_ab = ab - hip
#         ab_chest = chest - ab

#         # normalization
#         if np.linalg.norm(hip_ab) > 0 and np.linalg.norm(ab_chest) > 0:
#             angle_segments_raw = calculate_angle(hip_ab, ab_chest)
#             angle_segments = 180 - angle_segments_raw  # Complementary angle
#         else:
#             angle_segments = None  # Handle invalid vectors

#         # hip-chest segment relative to the ground (z-axis)
#         hip_chest = chest - hip
#         ground_vector = np.array([0, 0, 1])
#         if np.linalg.norm(hip_chest) > 0:
#             angle_ground = calculate_angle(hip_chest, ground_vector)
#         else:
#             angle_ground = None

#         angles.append({
#             "frame": idx,
#             "angle_segments": angle_segments,
#             "angle_ground": angle_ground
#         })

#         if idx < 10:  # Solo per i primi 10 frame
#             print(f"Frame {idx}:")
#             print(f"  hip_ab: {hip_ab}, ab_chest: {ab_chest}")
#             print(f"  Angle between segments (raw): {angle_segments_raw}")
#             print(f"  Angle between segments (corrected): {angle_segments}")

#     return pd.DataFrame(angles)


# Analyze spine oscillations
# oscillations_1, stats_1 = analyze_spine_oscillations(df_1)
# oscillations_2, stats_2 = analyze_spine_oscillations(df_2)
# ab_oscillations_1, stats_ab_1 = calculate_ab_statistics(df_1)
# ab_oscillations_2, stats_ab_2 = calculate_ab_statistics(df_2)
# alignment_angles_1 = calculate_spine_alignment(df_1)
# alignment_angles_2 = calculate_spine_alignment(df_2)
