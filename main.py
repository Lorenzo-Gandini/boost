import numpy as np
import optitrack.csv_reader as csv
from optitrack.geometry import *
from utils import *

# Load csv
filename = "lab_records/Lorenzo_1.csv"
take = csv.Take().readCSV(filename)

# Get the position of all joints and give them a color
body_edges, bones_pos, colors = get_bones_position(take)
angles = calculate_angles(bones_pos)
save_angles(angles)
show_points(take, bones_pos, body_edges, colors)
save_spine_values(bones_pos)

