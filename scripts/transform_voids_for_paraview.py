"""
Transform voids data for ParaView visualization.

This script converts void catalog data into a format suitable for 3D visualization
in ParaView, including void properties for filtering and analysis.
"""

import numpy as np
import pyvista as pv
from pathlib import Path

DATA_FILE = Path("data/desi/edr/processed/voidfinder/survey_aware_voids.txt")
OUT_FILE = Path("voids_paraview.vtp")

# Load the void data computed from previous analysis
data = np.loadtxt(DATA_FILE)

x, y, z = data[:, 0], data[:, 1], data[:, 2]
radius = data[:, 3]
volume = data[:, 4]
n_points = data[:, 5]
mean_density = data[:, 6]

# Create point cloud for ParaView visualization
points = np.column_stack((x, y, z))
point_cloud = pv.PolyData(points)

# Add properties for filtering in ParaView
# This enables comparison of different void characteristics
point_cloud["radius"] = radius
point_cloud["volume"] = volume
point_cloud["n_points"] = n_points
point_cloud["mean_density"] = mean_density

# Save the file for ParaView
point_cloud.save(OUT_FILE)
print(f"ParaView file saved: {OUT_FILE}")
