from astar_path import AStar
from grid_loader import Grid
from matplotlib import pyplot as plt

img_path = "./data/circuit.png" # Grid path
grid_res = 0.05 # m / pixel
start = (6, 3)
end = (24, 25)
# Load the Grid object
grid_obj = Grid()
grid_obj.load_from_image(img_path, grid_res)
plt.figure()
plt.imshow(grid_obj.grid_data.T, cmap=plt.cm.gray_r, origin='lower', 
            extent=[0, grid_obj.w_m, 0, grid_obj.h_m])
# Find Astar path from start to end
astar_planner = AStar()
astar_planner.load_grid(grid_obj)
astar_path_m = astar_planner.get_route(start, end)
# Visualize everything
plt.figure()
plt.imshow(grid_obj.grid_data.T, cmap=plt.cm.gray_r, origin='lower', 
            extent=[0, grid_obj.w_m, 0, grid_obj.h_m])
plt.plot(start[0], start[1], 'g+', markersize=10)
plt.plot(end[0], end[1], 'r+', markersize=10)
plt.plot(astar_path_m[:, 0], astar_path_m[:, 1], 'r.')
plt.show()
