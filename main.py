import optitrack.csv_reader as csv
from optitrack.geometry import *
from utils import *
from config import ATHLETE_MOD, MOCAP_RECORD

filename = f"lab_records/{ATHLETE_MOD}_{MOCAP_RECORD}.csv"
print(filename)

# Load csv
take = csv.Take().readCSV(filename)

# Get the position of all joints and give them a color
body_edges, bones_pos, colors = get_bones_position(take)

# angles = calculate_angles(bones_pos)
# save_angles(angles)
# save_spine_values(bones_pos)

# show_animation(take, bones_pos, body_edges, colors)
show_animation(take, bones_pos, body_edges, colors, [1, 14, 15])