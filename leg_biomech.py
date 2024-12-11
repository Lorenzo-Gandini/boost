import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import interpolate_invalid_values

file_1 = "output/angles.json"
file_2 = "output/angles_2.json"

interpolate_invalid_values(file_1)
interpolate_invalid_values(file_2)

# Load files
with open(file_1) as f:
    data1 = json.load(f)

with open(file_2) as f:
    data2 = json.load(f)

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

angles = ["knee_l", "knee_r", "ankle_l", "ankle_r"]

# stats for each angle
comparison_output = {}
for angle in angles:
    mean1, mean2 = df1[angle].mean(), df2[angle].mean()
    std1, std2 = df1[angle].std(), df2[angle].std()
    min1, min2 = df1[angle].min(), df2[angle].min()
    max1, max2 = df1[angle].max(), df2[angle].max()
    range1, range2 = max1 - min1, max2 - min2
    
    comparison_output[angle.upper()] = {
        "mean": f"Before: {mean1:.2f} - After: {mean2:.2f}",
        "std": f"Before:{std1:.2f} - After: {std2:.2f}",
        "min": f"Before:{min1:.2f} - After: {min2:.2f}",
        "max": f"Before:{max1:.2f} - After: {max2:.2f}",
        "range": f"Before:{range1:.2f} - After: {range2:.2f}"
    }

# print
for angle, metrics in comparison_output.items():
    print(f"{angle}:")
    for metric, values in metrics.items():
        print(f"  {metric}: {values}")
    print()


#BOXPLOT
df1["setting"] = "Before"
df2["setting"] = "After"
combined = pd.concat([df1, df2])

# Boxplot
combined.melt(id_vars="setting", value_vars=["knee_l", "knee_r", "ankle_l", "ankle_r"]).boxplot(
    by=["variable", "setting"], figsize=(10, 6)
)
plt.title("Angle Distribution by Setting")
plt.ylabel("Angle (degrees)")
plt.show()
