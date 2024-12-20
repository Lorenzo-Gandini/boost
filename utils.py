import numpy as np
import pandas as pd
import open3d as o3d
import os
import time
import json
from config import ATHLETE_MOD, ATHLETE_MOD_UC, MOCAP_RECORD

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

#--- LEGS ---#
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
    
def save_angles(angles):
    """
    Save values into a json
    """
    output_folder = f"output/LEGS/{ATHLETE_MOD}/json/"
    filename = f"{output_folder}{ATHLETE_MOD_UC}_angle_{MOCAP_RECORD}.json"
    os.makedirs(output_folder, exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(angles, f, indent=4)
    print(f"Angoli salvati in: {filename}")

#--- SPINE BACK ---#
def save_spine_values(bones_pos):
    """
    Extract metrics for the spine and save the json
    """
    spine_metrics = []

    for frame_idx, joints in enumerate(bones_pos):
        # Extract spine positions
        hip = joints[0].tolist()
        ab = joints[1].tolist()
        chest = joints[2].tolist()

        # Append metrics for the current frame
        spine_metrics.append({
            "frame": frame_idx,
            "hip": hip,
            "ab": ab,
            "chest": chest
        })

    output_folder = f"output/LEGS/{ATHLETE_MOD}/json/"
    filename = f"{output_folder}{ATHLETE_MOD_UC}_spine_metrics_{MOCAP_RECORD}.json"
    os.makedirs(output_folder, exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(spine_metrics, f, indent=4)
    print(f"Spine metrics saved to: {filename}")

def load_spine_data(file_path):
    """
    Load spine metrics from a JSON file and convert to a DataFrame.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    frames = []
    for frame_data in data:
        frame = {
            "frame": frame_data["frame"],
            "hip_x": frame_data["hip"][0],
            "hip_y": frame_data["hip"][1],
            "hip_z": frame_data["hip"][2],
            "ab_x": frame_data["ab"][0],
            "ab_y": frame_data["ab"][1],
            "ab_z": frame_data["ab"][2],
            "chest_x": frame_data["chest"][0],
            "chest_y": frame_data["chest"][1],
            "chest_z": frame_data["chest"][2],
        }
        frames.append(frame)

    return pd.DataFrame(frames)

#--- ANIMATION ---#
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


