import numpy as np
import pandas as pd
import open3d as o3d
import os
import time
import json
from optitrack.geometry import *
from scipy.signal import find_peaks

from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# ---- COMMUNICATIONS WITH THE USER.
# Set of funcions used to communicate with the user and ask for inputs.
def ask_athlete(prompt):
    """
    Ask the user to select an athlete from the list of athlets inside the folder training_data.
    """
    folder_path = "training_data/"
    athletes = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    print(f"🤔 {prompt}")
    for index, athlete in enumerate(athletes, start=1):
        print(f"   {index}. {athlete}")

    while True:
        try:
            choice = int(input("➡️  "))
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
    """
    print(f"🤔 {prompt}")
    print("   1. Spine movements")
    print("   2. Knee angles")
    print("   3. Ankle angles")
    print("   4. Training sessions")
    print("   5. All of them.")

    while True:
        try:
            reply = int(input("➡️  "))
            if reply in {1, 2, 3, 4, 5}:
                print(f"✅ You selected option {reply}\n")
                return reply
            else:
                raise ValueError
        except ValueError:
            print("❌ Invalid input. Please select a valid option (1-5).")

def ask_yesno(prompt):
    """
    Ask the user a yes/no question
    """
    valid_yes = {'y', 'Y', 'yes', 'YES', 'Yes'}
    valid_no = {'n', 'N', 'no', 'NO', 'No'}

    print(f"🤔 {prompt} (yes/no)")
    while True:
        option = input("➡️  ").strip()
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
    Ask the user which side of a joint they want to analyze
    """
    valid_inputs = {
        "r": "right", "right": "right", "1": "right",
        "l": "left", "left": "left", "2": "left",
        "b": "both", "both": "both", "3": "both",
    }
    print(f"🤔 Which {joint_name} do you want to analyze?")
    print("   1. Right\n   2. Left\n   3. Both")

    while True:
        side = input("➡️  ").strip().lower()
        if side in valid_inputs:
            print(f"✅ You selected: {valid_inputs[side]}")
            return valid_inputs[side]
        print("❌ Invalid input. Please type 'left', 'right', or 'both'.")

def user_message(message, message_type="info"):
    """
    Display a standardized message to the user with optional emojis to make the communications more enjoyable.
    """
    emoji_map = {
        "question": "❓",
        "saving_graph": "📶",
        "saving_stats": "💾",
        "info": "❗",
        "error": "❌",
        "success": "✅"
    }
    emoji = emoji_map.get(message_type, "ℹ️")
    if emoji == "success" or emoji == "question":
        print(f"\n{emoji} {message}")
    else:
        print(f"{emoji} {message}")

def print_recap(choices):
    """
    Print a recap of the user's choices at the end of the analysis.
    """
    print("---- RECAP ----")
    print(f"   Athlete: {choices.get('athlete', 'N/A')}")
    print(f"   Spine Analysis: {'Enabled' if choices.get('spine') else 'Disabled'}")
    print(f"   Leg Analysis: {'Enabled' if choices.get('leg') else 'Disabled'}")
    print(f"   Ankle Analysis: {'Enabled' if choices.get('ankle') else 'Disabled'}")
    print(f"   Training Analysis: {'Enabled' if choices.get('training') else 'Disabled'}")
    # print(f"   PDF Report: {'Yes' if choices.get('pdf') else 'No'}")
    print()

#--- GENERAL OPTIONS ---#
def get_bones_position(file):
    """
    Get the bones positions from the loaded csv file from the mocap.
    """
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

    #interpolate zeros when there are missing values
    bones_pos = interpolate_bones_positions(bones_pos)

    return body_edges, bones_pos, colors

def show_animation(file, bones_pos, body_edges, colors, points_indices=None):
    """
    Show the animation of the stickman. Optionally, highlight trajectories of specific giving points points.
    This function were used in first versions, now it's not used anymore but usefull if desired to show the animation.
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

def save_stats(stats1, stats2, athlete, joint, side=None):
    """
    Save the extracted statistics to a JSON file.
    """
    output_folder = f"output/{athlete}/stats/"
    os.makedirs(output_folder, exist_ok=True)
    if side == None:
        output_file = os.path.join(output_folder, f"{athlete}_{joint}_stats.json")
    else:
        output_file = os.path.join(output_folder, f"{athlete}_{joint}_{side}_stats.json")

    stats = {
        "Setting_1": stats1,
        "Setting_2": stats2
    }

    with open(output_file, "w") as f:
        json.dump(stats, f, indent=4)
    user_message(f"All stats for the {side} {joint} analysis have been saved in {output_file}. \n", "saving_stats")

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
                valid_mask = ~invalid_frames        #~ è operatore NOT che agisce su array
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
    Split the data into three temporal zones (assuming 30 FPS):
    - Zone 2: First 2 minutes.
    - Zone 3: Between 2 and 4 minutes.
    - Zone 5: Last minute of the recording.
    """
    fps = 30  # Assuming 30 frames per second
    zone_2 = df.iloc[:2 * 60 * fps]
    zone_3 = df.iloc[2 * 60 * fps:4 * 60 * fps]
    zone_5 = df.iloc[-1 * 60 * fps:]
    return zone_2, zone_3, zone_5

# --- ANALYSIS FUNCTIONS ---
# Functions used to analyze the diffeent body parts and extract the statistics.
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

def analyze_angle(df, point1_start, point1_end, point2_start, point2_end, angle_name):
    """
    Analyze the angle between two given vectors.
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
    Plot 2x3 histograms and distributions for the two settings and the different zones.
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
    Calculate the oscillations of the hip-abdomen vector projected on the hip-chest axis.
    """

    hip_idx = 0  # Hip index
    chest_idx = 2  # Chest index
    ab_idx = 1  #  Abdomen index

    # hip-chest axis
    hip_chest_vectors = bones_pos[:, chest_idx] - bones_pos[:, hip_idx]  
    hip_chest_mean = hip_chest_vectors.mean(axis=0)  
    hip_chest_mean_unit = hip_chest_mean / np.linalg.norm(hip_chest_mean)  

    #Oscillations are defined as the deviation of the hip-ab vector from the hip-chest axis (which is the mean vector)
    oscillations = []
    for frame in bones_pos:
        hip = frame[hip_idx]
        chest = frame[chest_idx]
        ab = frame[ab_idx]
        hip_ab = ab - hip

        projection_length = np.dot(hip_ab, hip_chest_mean_unit)
        projection = projection_length * hip_chest_mean_unit
        deviation = hip_ab - projection

        ortho_vector = np.cross(hip_chest_mean_unit, [0, 0, 1])  # Cross product con z-axis
        ortho_vector_unit = ortho_vector / np.linalg.norm(ortho_vector)

        deviation_magnitude = np.dot(deviation, ortho_vector_unit)

        oscillations.append(deviation_magnitude)

    oscillations = np.array(oscillations)
    oscillations_centered = oscillations - oscillations.mean()

    global_stats = {
        "mean": float(oscillations_centered.mean()),
        "std_dev": float(oscillations_centered.std()),
        "mean_left": float(oscillations_centered[oscillations_centered > 0].mean()),
        "mean_right": float(oscillations_centered[oscillations_centered < 0].mean())
    }

    # Save info for each zone
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
    """
    Plot the oscillations for the two settings.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharey=True)
    fig.suptitle(title)

    # Define temporal zones
    zone_limits = [0, 2 * 60 * 30, 4 * 60 * 30, len(data1)] 
    zone_colors = ['lightgreen', 'lightblue', 'lightcoral']  

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


def analyze_joint_angle(angles, angle_key):
    """
    Analyze a specific knee angle, returning stats about it
    """
    angle_data = np.array([frame[angle_key] for frame in angles])

    # Peaks and valleys (max and min)
    peaks, _ = find_peaks(angle_data, prominence=5)
    valleys, _ = find_peaks(-angle_data, prominence=5)

    # Cycles 
    num_cycles = min(len(peaks), len(valleys))
    peaks = peaks[:num_cycles]
    valleys = valleys[:num_cycles]

    cycle_amplitudes = angle_data[peaks] - angle_data[valleys]
    cycle_durations = np.diff(peaks)
    cadence = 60 / (np.mean(cycle_durations) / 30) if len(cycle_durations) > 0 else 0  

    # Angular velcoity
    angular_velocity = np.gradient(angle_data, 1 / 30)  
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
    Save and show (if user want it) the histogram and the distribution of angles for peaks or valleys.
    """
    def calculate_distribution(data, indices):
        """
        Function that extract the distribution, specific for this specific task
        """
        filtered_data = data[indices]
        kde = gaussian_kde(filtered_data)
        x = np.linspace(filtered_data.min(), filtered_data.max(), 500)
        y = kde(x)
        return {"mean": filtered_data.mean(), "std": filtered_data.std(), "x": x, "y": y}

    #Two distributions
    dist1 = calculate_distribution(angles1, indices1)
    dist2 = calculate_distribution(angles2, indices2)

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
    if show_plots:
        plt.show()
    plt.close()


def plot_polar_angles(indices1, indices2, angles1, angles2, title, label1, label2, output_file, show_plots):
    """
    Save and show (if user want it) the plot of angles in polar coordinates.
    """
    angles1_rad = np.deg2rad(angles1[indices1])
    angles2_rad = np.deg2rad(angles2[indices2])

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Plot the data
    ax.scatter(angles1_rad, indices1, color="red", label=label1)
    ax.scatter(angles2_rad, indices2, color="blue", label=label2)

    plt.title(title)
    plt.legend()

    # Save and optionally show the plot
    plt.savefig(output_file)
    if show_plots:
        plt.show()
    plt.close()

def plot_angle_and_velocity_for_cycle(angles1, angles2, peaks1, peaks2, cycle_index, title, label1, label2, output_file, show_plots, frame_rate=30):
    """
    Plot angle and angular velocity for a specific cycle for two settings. 
    """
    # Data for first setting
    start_frame1 = peaks1[cycle_index]
    end_frame1 = peaks1[cycle_index + 1]
    angles_cycle1 = angles1[start_frame1:end_frame1]
    velocity_cycle1 = np.gradient(angles_cycle1, 1 / frame_rate)

    # Data for second setting
    start_frame2 = peaks2[cycle_index]
    end_frame2 = peaks2[cycle_index + 1]
    angles_cycle2 = angles2[start_frame2:end_frame2]
    velocity_cycle2 = np.gradient(angles_cycle2, 1 / frame_rate)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(angles_cycle1, label=f"{label1} - Angle", color="red", linestyle="-")
    ax1.plot(angles_cycle2, label=f"{label2} - Angle", color="blue", linestyle="-")
    ax1.set_xlabel("Frame (within cycle)")
    ax1.set_ylabel("Angle (degrees)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left")
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.plot(velocity_cycle1, label=f"{label1} - Angular Velocity", color="red", linestyle="--")
    ax2.plot(velocity_cycle2, label=f"{label2} - Angular Velocity", color="blue", linestyle="--")
    ax2.set_ylabel("Angular Velocity (degrees/second)", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.legend(loc="upper right")

    # Save and optionally show the plot
    plt.title(title)
    plt.savefig(output_file)
    if show_plots:
        plt.show()
    plt.close()