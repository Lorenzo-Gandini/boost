import numpy as np
import pandas as pd
import open3d as o3d
import time
import json

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
    Interpolate missing or invalid bone positions (all zeros) in bones_pos.
    
    Parameters:
        bones_pos (np.ndarray): Array of joint positions (frames x joints x 3).
        
    Returns:
        np.ndarray: Interpolated bones_pos.
    """
    interpolated_bones = bones_pos.copy()
    num_frames, num_joints, _ = bones_pos.shape

    for joint in range(num_joints):
        # Identify invalid frames (all coordinates are zero)
        invalid_frames = (bones_pos[:, joint] == 0).all(axis=1)

        for dim in range(3):  # Iterate over x, y, z coordinates
            # Extract the coordinate values for the current joint and dimension
            joint_coord = bones_pos[:, joint, dim]

            if np.any(invalid_frames):
                # Create a valid mask
                valid_mask = ~invalid_frames
                
                # Interpolate only if there are valid points
                if np.any(valid_mask):
                    valid_indices = np.where(valid_mask)[0]
                    invalid_indices = np.where(invalid_frames)[0]

                    # Interpolate using NumPy's interpolation function
                    joint_coord[invalid_frames] = np.interp(
                        invalid_indices,
                        valid_indices,
                        joint_coord[valid_mask]
                    )
            
            # Update the interpolated_bones array for the current dimension
            interpolated_bones[:, joint, dim] = joint_coord

    return interpolated_bones

def show_points(file, bones_pos, body_edges, colors):
    
    # Create a point cloud for joints
    keypoints = o3d.geometry.PointCloud()
    keypoints.points = o3d.utility.Vector3dVector(bones_pos[0])         # Assign the positions of joints the first frame

    skeleton_joints = o3d.geometry.LineSet()                            # Create a LineSet for skeletal connections
    skeleton_joints.points = o3d.utility.Vector3dVector(bones_pos[0])   # Assign positions of joints
    # center_skel = skeleton_joints.get_center()                          # Compute the center of the skeletal structure
    skeleton_joints.lines = o3d.utility.Vector2iVector(body_edges)      # Define the connections between joints
    skeleton_joints.colors = o3d.utility.Vector3dVector(colors)         # Assign colors to the skeletal connections

    vis = o3d.visualization.Visualizer()    # Initialize the Open3D visualizer
    vis.create_window()                     # Create the visualization window

    vis.add_geometry(skeleton_joints)   # Add skeletal connections
    vis.add_geometry(keypoints)         # Add the point cloud representing joints

    # Convert Open3D point cloud to a NumPy array for further processing
    # points_np = np.asarray(keypoints.points)

    # Settings for the video
    duration = 15 #seconds
    frame_rate = file.frame_rate
    frame_count = int(frame_rate * duration)  
    interval = 1 / frame_rate  

    for i in range(frame_count):
        new_joints = bones_pos[i]
        # center_skel = skeleton_joints.get_center()
        skeleton_joints.points = o3d.utility.Vector3dVector(new_joints)
        keypoints.points = o3d.utility.Vector3dVector(new_joints)

        # Update the geometries of the skeleton
        vis.update_geometry(skeleton_joints)
        vis.update_geometry(keypoints)
        
        vis.update_renderer()
        vis.poll_events()

        time.sleep(interval)
        
    vis.run()


#--- LEGS ---#
def calculate_angles(bones_pos):
    """
    Calcola gli angoli per tutti i frame e li restituisce come lista di dizionari.
    
    Parameters:
        bones_pos (np.ndarray): Array delle posizioni dei joints (frames x joints x 3).
        
    Returns:
        list: Lista di dizionari contenenti gli angoli per ciascun frame.
    """
    angles = []

    for frame_idx, joints in enumerate(bones_pos):
        # extract angles for the specific frame
        knee_l_angle = calculate_angle(joints[13] - joints[14], joints[15] - joints[14])  # Ginocchio sinistro
        knee_r_angle = calculate_angle(joints[17] - joints[18], joints[19] - joints[18])  # Ginocchio destro
        ankle_l_angle = calculate_angle(joints[14] - joints[15], joints[16] - joints[15])  # Caviglia sinistra
        ankle_r_angle = calculate_angle(joints[18] - joints[19], joints[20] - joints[19])  # Caviglia destra

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
    Calcola l'angolo tra due vettori in gradi.
    """
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

        # Check for zero-length vectors
    if v1_norm == 0 or v2_norm == 0:
        return 0.0  # Default angle for invalid vectors

    dot_product = np.dot(v1, v2)
    # Clamp the value to avoid numerical issues with acos
    cos_theta = np.clip(dot_product / (v1_norm * v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg
    
def save_angles(angles):
    """
    Salva gli angoli calcolati in un file JSON.
    """
    filename="output/angles_1.json"
    with open(filename, 'w') as f:
        json.dump(angles, f, indent=4)
    print(f"Angoli salvati in: {filename}")

#--- SPINE BACK ---#
def save_spine_values(bones_pos):
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

    filename = "output/spine_metrics_1.json"

    with open(filename, 'w') as f:
        json.dump(spine_metrics, f, indent=4)
    print(f"Spine metrics saved to: {filename}")

def load_spine_data(file_path):
    """
    Load spine metrics from a JSON file and convert to a DataFrame.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        pd.DataFrame: DataFrame with spine data, including x, y, z for each joint.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Flatten the JSON structure for Pandas
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
