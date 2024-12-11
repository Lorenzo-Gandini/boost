import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import open3d as o3d
import time
import numpy as np

# Load the Optitrack CSV file parser module.
import optitrack.csv_reader as csv
from optitrack.geometry import *

# Load csv
filename = "lab_records/Lorenzo_1_20-11-2024.csv"
take = csv.Take().readCSV(filename)
# print("Found rigid bodies:", take.rigid_bodies.keys())

# Connect joints
bodies = take.rigid_bodies
body_edges = [[0,1],[1,2],[2,3],[3,4],[3,5],[5,6],[6,7],[7,8],[3,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]

bones_pos = []
if len(bodies) > 0:
    for body in bodies: 
        bones = take.rigid_bodies[body]
        # Ensure all frames have valid positions
        fixed_positions = [
            pos if pos is not None else [0.0, 0.0, 0.0]
            for pos in bones.positions
        ]
        bones_pos.append(fixed_positions)

bones_pos = np.array(bones_pos).transpose((1, 0, 2))
colors = [[1, 0, 0] for i in range(len(body_edges))]

keypoints = o3d.geometry.PointCloud()
keypoints.points = o3d.utility.Vector3dVector(bones_pos[0])
keypoints_center = keypoints.get_center()
keypoints.points = o3d.utility.Vector3dVector(bones_pos[0])
skeleton_joints = o3d.geometry.LineSet()
skeleton_joints.points = o3d.utility.Vector3dVector(bones_pos[0])
center_skel = skeleton_joints.get_center()
skeleton_joints.points = o3d.utility.Vector3dVector(bones_pos[0])
skeleton_joints.lines = o3d.utility.Vector2iVector(body_edges)
skeleton_joints.colors = o3d.utility.Vector3dVector(colors)

vis = o3d.visualization.Visualizer()
    
vis.create_window()

# Plot the entire skeleton
vis.add_geometry(skeleton_joints)
vis.add_geometry(keypoints)

points_np = np.asarray(keypoints.points)
frame_rate = take.frame_rate  
interval = 1 / frame_rate  

for i in range(1000,2003):
    new_joints = bones_pos[i]
    # center_skel = skeleton_joints.get_center()
    skeleton_joints.points = o3d.utility.Vector3dVector(new_joints)
    keypoints.points = o3d.utility.Vector3dVector(new_joints)
    
    # This plot the entire skeleton
    vis.update_geometry(skeleton_joints)
    vis.update_geometry(keypoints)
    
    vis.update_renderer()
    vis.poll_events()

    time.sleep(interval)
    
vis.run()