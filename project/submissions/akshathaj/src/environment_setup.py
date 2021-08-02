import numpy as np
import toml
from operator import itemgetter
import matplotlib.pyplot as plt
import copy

params = toml.load("config.toml")['params']

def grid_from_map(data,data_size):
	grid = np.zeros(data_size)
	for i in range(data_size[0]):
		for j in range(data_size[1]):
			if data[i][j] != '.':
				grid[i,j] = 1
	return grid
	
def place_obstacles(grid,obstacles,extent_limits):
	grid_copy = copy.deepcopy(grid)
	for obs in obstacles:
		# calculate obstacles extent in pixel coords
		ymin, ymax, xmax, xmin = obs
		# mark them as occupied
		xmax = extent_limits[-1]-xmax
		xmin = extent_limits[-1]-xmin
		grid_copy[xmin:xmax ,ymin:ymax ] = 1.0
	return grid_copy
		
def plot_environment(grid_dense,extent_limits):
	fig, ax = plt.subplots(figsize=(15,19))
	ax.imshow(np.flipud(grid_dense), cmap=plt.cm.Dark2,origin='Lower',extent=extent_limits)
	ax.grid()
	
def densify_grid(grid):
	grid_res = itemgetter('grid_res')(params)
	grid_size_x,grid_size_y = grid.shape
	grid_dense = np.zeros((int(grid_size_x/grid_res),int(grid_size_y/grid_res)))
	for x in range(grid_size_x):
		for y in range(grid_size_y):
			if grid[x,y] == 1:
				grid_dense[int(x/grid_res):int((x+1)/grid_res),int(y/grid_res):int((y+1)/grid_res)] = 1	
	return grid_dense