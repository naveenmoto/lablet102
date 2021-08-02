"""
    Testing DWA
    
    Current Notes:
    Forget the map for a while
"""

# %% Import everything
# Libraries
from grid_loader import Grid
from astar_path import AStar
from simulator import Lidar
from dwa import DWA
# Modules
import numpy as np
from matplotlib import pyplot as plt

# %% Global variables

V_MAX = 1.2 # Maximum velocity (m/s)
ACC_MAX = 0.5   # Maximum acceleration (m/s^2)
W_MAX = 1.0 # Maximum angular velocity (rad/sec)
W_DOT_MAX = 3.0 # Maximum angular acceleration (rad/sec^2)
K_MAX = 4 # Curvature = 1/R (m^-1)
pred_horizon = 10   # Predict N steps in DWA paths
goal_threshold = 0.05   # Goal threshold (m)
t_sim_end = 30 # Time for which simulation should run (s)
w_cte = 1.0 # Weight for cross track error in cost calculation
w_speed = 1.0   # Weight for speed change in cost calculation
w_kappa = 1e-4   # Cost of kappa value
w_lidar_cost = 0.0    # Cost of the position on grid
img_path = "./data/circuit.png" # Grid path
grid_res = 0.05 # m / pixel
start = (6, 3)
end = (24, 25)

# %% DWA implementation
dwa_handler = DWA(V_MAX, W_MAX, ACC_MAX, W_DOT_MAX, K_MAX)  # Paraemters
dwa_handler.set_path_props(pred_horizon, goal_threshold)
dwa_handler.set_weights(w_cte, w_speed, w_kappa, w_lidar_cost)

# %% Load grid and path
# Load grid and path
grid_obj = Grid()
grid_obj.load_from_image(img_path, grid_res)
# Astar path finding
astar_planner = AStar()
astar_planner.load_grid(grid_obj)
astar_path_m = astar_planner.get_route(start, end)
# Show start to end path
plt.figure()
plt.imshow(grid_obj.grid_data.T, cmap=plt.cm.gray_r, origin='lower', 
            extent=[0, grid_obj.w_m, 0, grid_obj.h_m])
plt.plot(start[0], start[1], 'g+', markersize=10)
plt.plot(end[0], end[1], 'r+', markersize=10)
plt.plot(astar_path_m[:, 0], astar_path_m[:, 1], 'r.')
plt.show()

# %% Run DWA on the 'astar_path_m' path
start_pose = (start[0], start[1], np.pi/2)  # Starting pose
ref_path = astar_path_m # Reference path
lidar_sim = Lidar()
lidar_sim.set_grid(grid_obj)

# Define the cost of being on path
def grid_cost(pose):
    """
        Estimate the cost of being at 'pose' on the
        grid

        Parameters:
        - pose: list
            Current pose of robot as (x, y, th)

        Returns:
        - cost: float
            The cost of being at pose on the grid
    """
    # Get the distances
    distances = np.array(lidar_sim.get_beam_data(pose))
    min_dist = np.min(distances)
    if min_dist < lidar_sim.scan_line_samples[-1]:
        cost = 1 / min_dist
    else:
        cost = 0
    return cost

# dwa_handler.pose_cost = grid_cost
logs = dwa_handler.run_dwa(start_pose, ref_path, t_sim_end)
print(f"Simulation ended in {logs[-1, -2]:.3f} seconds")

# %% Plot path and everything
poses = logs[:, :3]

plt.figure()
plt.imshow(grid_obj.grid_data.T, cmap=plt.cm.gray_r, origin='lower', 
            extent=[0, grid_obj.w_m, 0, grid_obj.h_m])
plt.plot(start[0], start[1], 'g+', markersize=10)
plt.plot(end[0], end[1], 'r+', markersize=10)
plt.plot(ref_path[:, 0], ref_path[:, 1], 'g.')
plt.plot(poses[:, 0], poses[:, 1], 'r')

# %%
