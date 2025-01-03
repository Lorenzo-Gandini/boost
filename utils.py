import numpy as np
import pandas as pd
import open3d as o3d
import os
import time
import json
from optitrack.geometry import *

from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

#-----MAIN----
def ask_athlete(prompt):
    """
    Ask the user to select an athlete from the list, with emoji-enhanced feedback.
    """
    folder_path = "training_data/"
    athletes = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    print(f"🤔 {prompt}")
    for index, athlete in enumerate(athletes, start=1):
        print(f"   {index}. {athlete}")

    while True:
        try:
            choice = int(input("Your choice: "))
            if 1 <= choice <= len(athletes):
                selected_athlete = athletes[choice - 1]
                print(f"✅ You selected: {selected_athlete}\n")
                return selected_athlete
            else:
                raise ValueError
        except ValueError:
            print("❌ Invalid input. Please select a valid number.")

def ask_option(prompt):
    """
    Ask the user to select an option for the type of analysis.
    Returns the selected option as an integer.
    """
    print(f"🤔 {prompt}")
    print("   1. Spine movements")
    print("   2. Knee angles")
    print("   3. Ankle angles")
    print("   4. Training sessions")
    print("   5. All of them.")

    while True:
        try:
            reply = int(input("Your choice: "))
            if reply in {1, 2, 3, 4, 5}:
                print(f"✅ You selected option {reply}\n")
                return reply
            else:
                raise ValueError
        except ValueError:
            print("❌ Invalid input. Please select a valid option (1-5).")



def ask_yesno(prompt):
    """
    Ask the user a yes/no question with emoji-enhanced feedback.
    """
    valid_yes = {'y', 'Y', 'yes', 'YES', 'Yes'}
    valid_no = {'n', 'N', 'no', 'NO', 'No'}

    print(f"🤔 {prompt} (yes/no)")
    while True:
        option = input("Your choice: ").strip()
        if option in valid_yes:
            print("✅ You selected: Yes\n")
            return True
        elif option in valid_no:
            print("✅ You selected: No\n")
            return False
        else:
            print("❌ Invalid input. Please respond with 'yes' or 'no'.")


def ask_joint_side(joint_name):
    """
    Ask the user which side of a joint they want to analyze (e.g., Right, Left, or Both).
    """
    valid_inputs = {
        "r": "right", "right": "right", "1": "right",
        "l": "left", "left": "left", "2": "left",
        "b": "both", "both": "both", "3": "both",
    }
    print(f"🤔 Which {joint_name} do you want to analyze?")
    print("   1. Right\n   2. Left\n   3. Both")

    while True:
        side = input("Your choice: ").strip().lower()
        if side in valid_inputs:
            print(f"✅ You selected: {valid_inputs[side]}\n")
            return valid_inputs[side]
        print("❌ Invalid input. Please type 'left', 'right', or 'both'.")

def user_message(message, message_type="info"):
    """
    Display a standardized message to the user with optional emojis.
    
    Parameters:
    - message: The text of the message.
    - message_type: The type of message ('question', 'saving_graph', 'saving_stats', 'info').
    
    Emoji Mapping:
    - 'question': 🤔
    - 'saving_graph': 📊
    - 'saving_stats': 💾
    - 'info': ℹ️
    """
    emoji_map = {
        "question": "🤔",
        "saving_graph": "📊",
        "saving_stats": "💾",
        "info": "ℹ️",
        "error": "❌",
        "success": "✅"
    }
    emoji = emoji_map.get(message_type, "ℹ️")
    print(f"{emoji} {message}")

def print_recap(choices):
    """
    Print a recap of the user's choices at the end of the analysis.
    
    Parameters:
    - choices: Dictionary containing user choices.
    """
    print("\n---- RECAP OF YOUR CHOICES ----")
    print(f"   Athlete: {choices.get('athlete', 'N/A')}")
    print(f"   SPINE Analysis: {'Enabled' if choices.get('spine') else 'Disabled'}")
    print(f"   LEG Analysis: {'Enabled' if choices.get('leg') else 'Disabled'}")
    print(f"   ANKLE Analysis: {'Enabled' if choices.get('ankle') else 'Disabled'}")
    print(f"   TRAINING Analysis: {'Enabled' if choices.get('training') else 'Disabled'}")
    print(f"   PDF Report: {'Yes' if choices.get('pdf') else 'No'}")
    print()


def show_animation(file, bones_pos, body_edges, colors, points_indices=None):
    """
    Show the animation of the stickman. Optionally, highlight trajectories of specific points.

    Parameters:
        file: Loaded file containing the frame rate.
        bones_pos: Positions of all bones over time.
        body_edges: Edges defining the connections between bones.
        colors: Colors for the edges.
        points_indices: Optional list of indices of the points whose trajectories should be shown.
                        If None, no trajectories are shown, only the stickman animation.
    """
    # Ensure points_indices is a list if provided
    if points_indices is not None and isinstance(points_indices, int):
        points_indices = [points_indices]

    # Create a point cloud for joints
    keypoints = o3d.geometry.PointCloud()
    keypoints.points = o3d.utility.Vector3dVector(bones_pos[0])

    # Create a LineSet for skeletal connections
    skeleton_joints = o3d.geometry.LineSet()
    skeleton_joints.points = o3d.utility.Vector3dVector(bones_pos[0])
    skeleton_joints.lines = o3d.utility.Vector2iVector(body_edges)
    skeleton_joints.colors = o3d.utility.Vector3dVector(colors)

    # Create LineSets for trajectories if points_indices is provided
    trajectories = {}
    if points_indices is not None:
        for idx in points_indices:
            trajectories[idx] = {
                "points": [],
                "lines": [],
                "geometry": o3d.geometry.LineSet()
            }

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(skeleton_joints)
    vis.add_geometry(keypoints)

    # Add all trajectories to the visualizer if points_indices is provided
    if points_indices is not None:
        for traj in trajectories.values():
            vis.add_geometry(traj["geometry"])

    # Settings for the animation
    frame_rate = file.frame_rate
    num_frames = bones_pos.shape[0]
    interval = 1 / frame_rate

    for i in range(num_frames):
        # Update skeleton and keypoints positions
        new_joints = bones_pos[i]
        skeleton_joints.points = o3d.utility.Vector3dVector(new_joints)
        keypoints.points = o3d.utility.Vector3dVector(new_joints)

        # Update trajectories for each point if points_indices is provided
        if points_indices is not None:
            for idx in points_indices:
                trajectories[idx]["points"].append(new_joints[idx])
                if len(trajectories[idx]["points"]) > 1:
                    trajectories[idx]["lines"].append([
                        len(trajectories[idx]["points"]) - 2,
                        len(trajectories[idx]["points"]) - 1
                    ])

                trajectory_geometry = trajectories[idx]["geometry"]
                trajectory_geometry.points = o3d.utility.Vector3dVector(np.array(trajectories[idx]["points"], dtype=np.float64))
                if trajectories[idx]["lines"]:
                    trajectory_geometry.lines = o3d.utility.Vector2iVector(np.array(trajectories[idx]["lines"], dtype=np.int32))

        # Update the visualizer
        vis.update_geometry(skeleton_joints)
        vis.update_geometry(keypoints)
        if points_indices is not None:
            for traj in trajectories.values():
                vis.update_geometry(traj["geometry"])

        vis.poll_events()
        vis.update_renderer()
        time.sleep(interval)

    vis.run()

def save_stats(stats1, stats2, athlete, athlete_mod_uc, joint, side):
    """
    Save the extracted statistics to a JSON file.
    """
    output_folder = f"output/{athlete}/stats/"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{athlete_mod_uc}_{joint}_{side}_stats.json")

    stats = {
        "Setting_1": stats1,
        "Setting_2": stats2
    }

    with open(output_file, "w") as f:
        json.dump(stats, f, indent=4)
    user_message(f"All stats for the {side} {joint} analysis have been saved.", "saving_stats")


#--- GENERAL OPTIONS ---#
def get_bones_position(file):
    bodies = file.rigid_bodies
    body_edges = [
        [0, 1],  # Hip -> Ab
        [1, 2],  # Ab -> Chest
        [2, 3],  # Chest -> Neck
        [3, 4],  # Neck -> Head
        [3, 5],  # Neck -> LShoulder
        [5, 6],  # LShoulder -> LUArm
        [6, 7],  # LUArm -> LFArm
        [7, 8],  # LFArm -> LHand
        [3, 9],  # Neck -> RShoulder
        [9, 10], # RShoulder -> RUArm
        [10, 11],# RUArm -> RFArm
        [11, 12],# RFArm -> RHand
        [0, 13], # Hip -> LThigh
        [13, 14],# LThigh -> LShin
        [14, 15],# LShin -> LFoot
        [15, 16],# LFoot -> LToe
        [0, 17], # Hip -> RThigh
        [17, 18],# RThigh -> RShin
        [18, 19],# RShin -> RFoot
        [19, 20] # RFoot -> RToe
    ]

    bones_pos = []
    if len(bodies) > 0:
        for body in bodies: 
            bones = file.rigid_bodies[body]
            
            # set 0,0,0 in case point missing
            fixed_positions = [
                pos if pos is not None else [0.0, 0.0, 0.0]
                for pos in bones.positions
            ]
            bones_pos.append(fixed_positions)

    bones_pos = np.array(bones_pos).transpose((1, 0, 2))
    colors = [[1, 0, 0] for i in range(len(body_edges))]

    #interpolate zeros
    bones_pos = interpolate_bones_positions(bones_pos)

    return body_edges, bones_pos, colors

def interpolate_bones_positions(bones_pos):
    """
    Interpolate invalid frames
    """
    interpolated_bones = bones_pos.copy()
    num_frames, num_joints, _ = bones_pos.shape

    for joint in range(num_joints):
        invalid_frames = (bones_pos[:, joint] == 0).all(axis=1) #find invalid (=0,0,0) frames
        
        for dim in range(3): #3=x,y,z
            joint_coord = bones_pos[:, joint, dim]  #extract the value for the current joint and dimension

            if np.any(invalid_frames):
                valid_mask = ~invalid_frames        #~ è operatore NOT che agisce su array, creiamo una maschera complementare
                if np.any(valid_mask):
                    valid_indices = np.where(valid_mask)[0]
                    invalid_indices = np.where(invalid_frames)[0]

                    # NumPy's interpolation function
                    joint_coord[invalid_frames] = np.interp(
                        invalid_indices,
                        valid_indices,
                        joint_coord[valid_mask]
                    )
            
            # Update the interpolated_bones array for the current dimension
            interpolated_bones[:, joint, dim] = joint_coord

    return interpolated_bones

def get_zones(df):
    """
    Split the data into three temporal zones:
    - Zone 2: First 2 minutes (assuming 30 FPS).
    - Zone 3: Between 2 and 4 minutes.
    - Zone 5: Last minute of the recording.
    """
    fps = 30  # Assuming 30 frames per second
    zone_2 = df.iloc[:2 * 60 * fps]
    zone_3 = df.iloc[2 * 60 * fps:4 * 60 * fps]
    zone_5 = df.iloc[-1 * 60 * fps:]
    return zone_2, zone_3, zone_5

#--- KNEE AND ANKLE ---#
def calculate_angles(bones_pos):
    """
    Give back a dictionary with all the angles of the legs
    """
    angles = []

    for frame_idx, joints in enumerate(bones_pos):
        # extract angles for the specific frame
        knee_l_angle = calculate_angle(joints[13] - joints[14], joints[15] - joints[14])    # ginocchio sinistro
        knee_r_angle = calculate_angle(joints[17] - joints[18], joints[19] - joints[18])    # inocchio destro
        ankle_l_angle = calculate_angle(joints[14] - joints[15], joints[16] - joints[15])   # caviglia sinistra
        ankle_r_angle = calculate_angle(joints[18] - joints[19], joints[20] - joints[19])   # caviglia destra

        # dictionary with the angles and the position
        angles.append({
            "frame": frame_idx,
            "knee_l": knee_l_angle,
            "knee_r": knee_r_angle,
            "ankle_l": ankle_l_angle,
            "ankle_r": ankle_r_angle,
        })
    
    return angles

def calculate_angle(v1, v2):
    """
    Angle between two vectors
    """
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    # in case one of vectors is 0, angle is 0
    if v1_norm == 0 or v2_norm == 0:
        return 0.0  

    dot_product = np.dot(v1, v2)
    
    # clip value to avoid numerical issues with acos
    cos_theta = np.clip(dot_product / (v1_norm * v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

#--- SPINE BACK ---#
def analyze_angle(df, point1_start, point1_end, point2_start, point2_end, angle_name):
    """
    Extract the angle between two given vectors.
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
            angle = 180 - calculate_angle(vec1, vec2)  # Complementary angle
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

def analyze_angle_from_points(bones_pos, point1_start_idx, point1_end_idx, point2_start_idx, point2_end_idx):
    """
    Extract the angle between two given vectors directly from bones_pos data.
    """
    angles = []
    ground_vector = np.array([0, 0, 1])  # Static ground vector for 'ground'

    for frame in bones_pos:
        start1 = np.array(frame[point1_start_idx])
        end1 = np.array(frame[point1_end_idx])

        if point2_start_idx == 'ground' or point2_end_idx == 'ground':
            vec1 = end1 - start1
            vec2 = ground_vector
        else:
            start2 = np.array(frame[point2_start_idx])
            end2 = np.array(frame[point2_end_idx])
            vec1 = end2 - start2
            vec2 = end1 - start1

        if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
            angle = 180 - calculate_angle(vec1, vec2)  # Complementary angle
            angles.append(angle)
        else:
            angles.append(None)

    angles = pd.Series(angles, name="Angle")

    stats = {
        'mean_angle': angles.mean(),
        'std_angle': angles.std(),
        'max_angle': angles.max(),
        'min_angle': angles.min()
    }

    return angles, stats

def plot_angles_combined(zones_1, zones_2, labels, title_prefix, output_file, show_plots):
    """
    Plot 2x3 histograms and KDE for two settings and multiple zones.
    """
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
        plt.ylim(0, 1.5)
        plt.xlim(*xlim)

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
        plt.ylim(0, 1.5)
        plt.xlim(*xlim)

    plt.tight_layout()
    plt.savefig(output_file)
    if show_plots:
        plt.show()
    plt.close()

def calculate_oscillations(bones_pos, zones=None):
    """
    Calcola le oscillazioni del punto 'ab' rispetto all'asse hip-chest.
    Ritorna:
    - Le oscillazioni nel tempo centrate rispetto alla media dell'asse hip-chest.
    - Le statistiche globali e per ciascuna zona, se specificata.
    """
    hip_idx = 0  # Indice per il punto Hip
    chest_idx = 2  # Indice per il punto Chest
    ab_idx = 1  # Indice per il punto Abdomen (Ab)

    # Calcolo dell'asse medio hip-chest
    hip_chest_vectors = bones_pos[:, chest_idx] - bones_pos[:, hip_idx]  # Vettori hip -> chest
    hip_chest_mean = hip_chest_vectors.mean(axis=0)  # Media su tutti i frame
    hip_chest_mean_unit = hip_chest_mean / np.linalg.norm(hip_chest_mean)  # Normalizzazione a vettore unitario

    # Calcolo delle oscillazioni
    oscillations = []
    for frame in bones_pos:
        hip = frame[hip_idx]
        chest = frame[chest_idx]
        ab = frame[ab_idx]

        # Vettore hip-ab
        hip_ab = ab - hip

        # Proiezione di hip-ab sul vettore medio hip-chest
        projection_length = np.dot(hip_ab, hip_chest_mean_unit)
        projection = projection_length * hip_chest_mean_unit

        # Deviazione perpendicolare
        deviation = hip_ab - projection

        # Direzione ortogonale rispetto all'asse hip-chest (vista dall'alto)
        ortho_vector = np.cross(hip_chest_mean_unit, [0, 0, 1])  # Cross product con z-axis
        ortho_vector_unit = ortho_vector / np.linalg.norm(ortho_vector)

        # Proiezione della deviazione sull'asse ortogonale
        deviation_magnitude = np.dot(deviation, ortho_vector_unit)

        # Salva il valore della deviazione nel tempo
        oscillations.append(deviation_magnitude)

    # Centralizza le oscillazioni rispetto al valore medio
    oscillations = np.array(oscillations)
    oscillations_centered = oscillations - oscillations.mean()

    # Calcola statistiche globali
    global_stats = {
        "mean": float(oscillations_centered.mean()),
        "std_dev": float(oscillations_centered.std()),
        "mean_left": float(oscillations_centered[oscillations_centered > 0].mean()),
        "mean_right": float(oscillations_centered[oscillations_centered < 0].mean())
    }

    # Calcola statistiche per ciascuna zona
    zone_names = ["Zone 2", "Zone 3", "Zone 5"]
    zone_stats = []
    if zones:
        for i, zone in enumerate(zones):
            zone_oscillations = oscillations_centered[zone.index]
            zone_stats.append({
                "zone": zone_names[i],
                "mean": float(zone_oscillations.mean()),
                "std_dev": float(zone_oscillations.std()),
                "mean_left": float(zone_oscillations[zone_oscillations > 0].mean()),
                "mean_right": float(zone_oscillations[zone_oscillations < 0].mean())
            })

    return oscillations_centered, {"global": global_stats, "zones": zone_stats}

def plot_oscillations(data1, data2, title, save_path, show_plots):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharey=True)
    fig.suptitle(title)

    # Define temporal zones
    zone_limits = [0, 2 * 60 * 30, 4 * 60 * 30, len(data1)]  # Frames: 0, 2 min, 4 min, end (assuming 30 FPS)
    zone_colors = ['lightgreen', 'lightblue', 'lightcoral']  # Colors for the zones

    # Plot for Setting 1
    for zone_start, zone_end, color in zip(zone_limits[:-1], zone_limits[1:], zone_colors):
        ax1.axvspan(zone_start, zone_end, color=color, alpha=0.3)  # Highlight zones
    ax1.plot(data1, label="Setting 1", color='blue')
    ax1.set_title("Setting 1")
    ax1.set_ylim(-10, 10)  # Set Y-axis range
    ax1.axhline(0, color='black', linestyle='--')  # Reference line
    ax1.legend()

    # Plot for Setting 2
    for zone_start, zone_end, color in zip(zone_limits[:-1], zone_limits[1:], zone_colors):
        ax2.axvspan(zone_start, zone_end, color=color, alpha=0.3)  # Highlight zones
    ax2.plot(data2, label="Setting 2", color='orange')
    ax2.set_title("Setting 2")
    ax2.set_ylim(-10, 10)  # Set Y-axis range
    ax2.axhline(0, color='black', linestyle='--')  # Reference line
    ax2.legend()

    # Save and show
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    if show_plots:
        plt.show()
    plt.close()

#--- COMMUNICATE ---#
def user_message(message, message_type="info"):
    """
    Display a standardized message to the user with optional emojis.
    
    Parameters:
    - message: The text of the message.
    - message_type: The type of message ('question', 'saving_graph', 'saving_stats', 'info').
    
    Emoji Mapping:
    - 'question': 🤔
    - 'saving_graph': 📊
    - 'saving_stats': 💾
    - 'info': ℹ️
    """
    emoji_map = {
        "question": "🤔",
        "saving_graph": "📊",
        "saving_stats": "💾",
        "info": "ℹ️",
        "error": "❌",
        "success": "✅"
    }
    emoji = emoji_map.get(message_type, "ℹ️")
    print(f"{emoji} {message}")

# WORKINPROGRESS 
def compute_symmetry(left_data, right_data, metric_name="Angle", show_plots=True):
    """
    Compute symmetry metrics for given left and right data, and optionally plot the results.
    
    Parameters:
        left_data (array-like): Data for the left side (e.g., left knee or ankle angles).
        right_data (array-like): Data for the right side (e.g., right knee or ankle angles).
        metric_name (str): Name of the metric being analyzed (e.g., "Angle", "Velocity").
        show_plots (bool): Whether to display plots for comparison.
    
    Returns:
        dict: Dictionary containing symmetry metrics.
    """
    # Compute basic statistics
    left_mean = np.mean(left_data)
    right_mean = np.mean(right_data)
    left_std = np.std(left_data)
    right_std = np.std(right_data)

    # Symmetry ratio and asymmetry index
    symmetry_ratio = left_mean / right_mean if right_mean != 0 else np.nan
    asymmetry_index = abs(left_mean - right_mean)

    # Package metrics
    metrics = {
        "left_mean": left_mean,
        "right_mean": right_mean,
        "left_std": left_std,
        "right_std": right_std,
        "symmetry_ratio": symmetry_ratio,
        "asymmetry_index": asymmetry_index
    }

    # Plot data
    if show_plots:
        plt.figure(figsize=(10, 5))
        plt.plot(left_data, label=f"Left {metric_name}", color="blue")
        plt.plot(right_data, label=f"Right {metric_name}", color="red")
        plt.title(f"{metric_name} Symmetry Analysis")
        plt.xlabel("Frames")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid()
        plt.show()

    return metrics

# Example usage:
# Replace these with the actual left and right data
left_knee_data = np.random.normal(45, 5, 100)  # Example data for left knee
right_knee_data = np.random.normal(43, 5, 100)  # Example data for right knee

symmetry_metrics = compute_symmetry(left_knee_data, right_knee_data, metric_name="Knee Angle", show_plots=True)
print("Symmetry Metrics:", symmetry_metrics)