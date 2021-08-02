import numpy as np
import networkx as nx

def euclidean(node1, node2):
    x1,y1 = node1
    x2,y2 = node2
    return 1*np.sqrt((x2-x1)**2 + (y2-y1)**2)
def initgraph(grid):
    #initialize graph
    grid_size = grid.shape
    G = nx.grid_2d_graph(*grid_size)
    deleted_nodes = 0 # counter to keep track of deleted nodes
    #nested loop to remove unconnected nodes
    for i in range(grid.shape[0]):
      for j in range(grid.shape[0]):
        if grid[i,j] == 1:
          G.remove_node((i,j))
          deleted_nodes += 1
    #print(f"removed {deleted_nodes} nodes")
    #print(f"number of occupied cells in grid {np.sum(grid)}")
    # pos = {(x,y):(y,-x) for x,y in G.nodes()}
    # nx.draw(G, pos = pos, node_color = 'red', node_size=2)
    return G

def astar_path(grid,start, goal):
    G = initgraph(grid)
    astar_path = nx.astar_path(G, start, goal, heuristic=euclidean, weight="weight")
    return astar_path