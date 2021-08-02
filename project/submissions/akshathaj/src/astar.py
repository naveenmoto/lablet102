import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def euclidean(node1, node2):
    x1,y1 = node1
    x2,y2 = node2
    dist = np.sqrt(pow((y2-y1),2) + pow((x2-x1),2))
    return dist
		  
def gen_astar_path(grid,start,goal):
	#initialize graph
	grid_path = np.flipud(grid)
	grid_size = grid_path.shape
	G = nx.grid_2d_graph(*grid_size)
			  
	nx.set_edge_attributes(G,1,'cost')
	G.add_edges_from([
		((x, y), (x+1, y+1))
		for x in range(grid_size[0]-1)
		for y in range(grid_size[1]-1)
	] + [
		((x+1, y), (x, y+1))
		for x in range(grid_size[0]-1)
		for y in range(grid_size[1]-1)
	], weight=1.4)
	
	for i in range(grid_size[0]):
		for j in range(grid_size[1]):
			if grid_path[i,j] == 1:
				G.remove_node((i,j))
	start_path = (start[1],start[0])
	goal_path = (goal[1],goal[0])
	astar_path = nx.astar_path(G,start_path,goal_path,heuristic=euclidean,weight='cost')
	astar_final = [(y,x) for x,y in astar_path]
	
	extent_limits = [0,grid_size[1],0,grid_size[0]]
	return astar_final,extent_limits
	
def plot_astar(grid,start,goal,astar_path,extent_limits):
	fig, ax = plt.subplots(figsize=(15,19))
	ax.imshow(np.flipud(grid), cmap=plt.cm.Dark2,origin='Lower',extent=extent_limits)
	ax.scatter(start[0],start[1], marker = "+", color = "yellow", s = 200)
	ax.scatter(goal[0],goal[1], marker = "+", color = "red", s = 200)
	for s in astar_path[1:]:
		ax.plot(s[0], s[1],'r+')	
	ax.grid()