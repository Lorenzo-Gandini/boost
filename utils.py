import numpy as np
import pandas as pd
import open3d as o3d
import time
import json

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
            
            # There is no position, set 0,0,0 
            fixed_positions = [
                pos if pos is not None else [0.0, 0.0, 0.0]
                for pos in bones.positions
            ]
            bones_pos.append(fixed_positions)

    bones_pos = np.array(bones_pos).transpose((1, 0, 2))
    
    # put a color to each dot
    colors = [[1, 0, 0] for i in range(len(body_edges))]

    return body_edges, bones_pos, colors

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
    filename="output/angles_2.json"
    with open(filename, 'w') as f:
        json.dump(angles, f, indent=4)
    print(f"Angoli salvati in: {filename}")

def show_points(file, bones_pos, body_edges, colors):
    
    # Create a point cloud for joints
    keypoints = o3d.geometry.PointCloud()
    keypoints.points = o3d.utility.Vector3dVector(bones_pos[0])         # Assign the positions of joints the first frame

    skeleton_joints = o3d.geometry.LineSet()                            # Create a LineSet for skeletal connections
    skeleton_joints.points = o3d.utility.Vector3dVector(bones_pos[0])   # Assign positions of joints
    center_skel = skeleton_joints.get_center()                          # Compute the center of the skeletal structure
    skeleton_joints.lines = o3d.utility.Vector2iVector(body_edges)      # Define the connections between joints
    skeleton_joints.colors = o3d.utility.Vector3dVector(colors)         # Assign colors to the skeletal connections

    vis = o3d.visualization.Visualizer()    # Initialize the Open3D visualizer
    vis.create_window()                     # Create the visualization window

    # Add geometries to the visualizer
    vis.add_geometry(skeleton_joints)   # Add skeletal connections
    vis.add_geometry(keypoints)         # Add the point cloud representing joints

    # Convert Open3D point cloud to a NumPy array for further processing
    points_np = np.asarray(keypoints.points)

    # Settings for the video
    duration = 5
    frame_rate = file.frame_rate
    frame_count = int(frame_rate * duration)  
    interval = 1 / frame_rate  

    for i in range(frame_count):
        new_joints = bones_pos[i]
        center_skel = skeleton_joints.get_center()
        skeleton_joints.points = o3d.utility.Vector3dVector(new_joints)
        keypoints.points = o3d.utility.Vector3dVector(new_joints)

        # Update the geometries of the skeleton
        vis.update_geometry(skeleton_joints)
        vis.update_geometry(keypoints)
        
        vis.update_renderer()
        vis.poll_events()

        time.sleep(interval)
        
    vis.run()

# Function to interpolate invalid (zero) values in a JSON file
def interpolate_invalid_values(file_path):
    # Load the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert JSON to DataFrame
    df = pd.DataFrame(data)
    
    # List of angles to interpolate
    angles = ["knee_l", "knee_r", "ankle_l", "ankle_r"]
    
    for angle in angles:
        # Get the indices of invalid (zero) values
        invalid_indices = df[df[angle] == 0].index
        
        for idx in invalid_indices:
            # Find previous and next valid indices
            prev_idx = idx - 1
            next_idx = idx + 1
            
            while next_idx < len(df) and df.at[next_idx, angle] == 0:
                next_idx += 1
            
            # If both previous and next valid values exist, interpolate
            if 0 <= prev_idx < len(df) and next_idx < len(df):
                prev_value = df.at[prev_idx, angle]
                next_value = df.at[next_idx, angle]
                df.at[idx, angle] = (prev_value + next_value) / 2
    
    # Save the updated DataFrame back to the JSON file
    with open(file_path, 'w') as f:
        json.dump(df.to_dict(orient='records'), f, indent=4)
    
    print(f"Interpolated invalid values and saved to {file_path}")
