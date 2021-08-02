import numpy as np
import networkx as nx


def euclidean(node1, node2):
    x1, y1 = node1
    x2, y2 = node2
    return np.sqrt((x1-x2)**2+(y1-y2)**2)


class a_star(object):

  def __init__(self,grid,threshold):
    self.grid=grid
    self.threshold=threshold
  #create a 2d graph for the grid
    self.dim=self.grid.shape
    self.G=nx.grid_2d_graph(self.dim[0],self.dim[1])

  def graph_from_grid(self):
    #remove nodes whose greyscale value is below threshold.As lower grayscale means its in the black area which means its outside the path
    node_removed=0
    for i in range(0,self.dim[0]):
      for j in range(0,self.dim[1]):
        if self.grid[i,j]<self.threshold:
          self.G.remove_node((i,j))
          node_removed+=1
    return self.G
  
  def search(self,start,goal):
    nx.set_edge_attributes(self.G,1, name="weight")
    astar_path =nx.astar_path(self.G, start, goal, heuristic=euclidean, weight='weight')
    return astar_path
