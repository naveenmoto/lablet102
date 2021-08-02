
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Planner():
    def __init__(self,grid):
        
        self.grid = grid # In grin, cells are classified as: 1 - occupied 0- free
        self.nodes = self.grid.shape        
        
        #initialize graph
        self.G = nx.grid_2d_graph(self.nodes[0],self.nodes[1])

        self.deleted_nodes = 0 # counter to keep track of deleted nodes

        nx.set_edge_attributes(self.G,1,name='cost')
        self.delete_node() #all the predefined obstacles are remved grom the networkx graph such that astar path can avoid these known obstacles
    
    def delete_node(self):
        for i in range(self.nodes[0]):
            for j in range(self.nodes[1]):
                if (self.grid[i,j]==1) and ((i,j) in self.G.nodes):
                    self.G.remove_node((i,j))
                    self.deleted_nodes+=1
        print(f"removed {self.deleted_nodes} nodes")
        print(f"number of occupied cells in grid {np.sum(self.grid)}")

    def compute_path(self,start,goal):
        astar_path = nx.astar_path(self.G,start,goal,heuristic=euclidean,weight='cost')
        x=[]
        y =[]
        for i in range(len(astar_path)-1): #this loop refines the astar_path by addnig midpoints between two points as also a point tin the reference astar path
            s = astar_path[i]
            r = astar_path[i+1]
            x.append(s[0])
            x.append((s[0]+r[0])/2)
            y.append(s[1])
            y.append((s[1]+r[1])/2)
        return x,y

def euclidean(node1, node2):
    '''
    calculates euclidean distance between two nodes
    '''
    x1,y1 = node1
    x2,y2 = node2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)
