from simulator import Lidar
from grid_loader import Grid
from astar_path import AStar

from matplotlib import pyplot as plt
import numpy as np

img_path = "./data/circuit.png" # Grid path
grid_res = 0.05 # m / pixel
pose = (7.5, 5, np.deg2rad(10))
# Load the Grid object
grid_obj = Grid()
grid_obj.load_from_image(img_path, grid_res)
plt.figure()
plt.imshow(grid_obj.grid_data.T, cmap=plt.cm.gray_r, origin='lower', 
            extent=[0, grid_obj.w_m, 0, grid_obj.h_m])

# Simulate LiDAR
lidar_sim = Lidar()
lidar_sim.set_grid(grid_obj)
dist = lidar_sim.get_beam_data(pose)
print(f"Distances are {dist}")

# Show the result
plt.figure()
plt.imshow(grid_obj.grid_data.T, cmap=plt.cm.gray_r, origin='lower', 
            extent=[0, grid_obj.w_m, 0, grid_obj.h_m])
cx, cy, cth = pose
for d, t in zip(dist, lidar_sim.scan_angles):
    theta = cth + t
    plt.plot((cx, cx+d*np.cos(theta)), (cy, cy+d*np.sin(theta)), 'r-')
plt.show()
