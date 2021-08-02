"""
    Generates a path on the given occupancy grid (map of
    the environment)
"""

import networkx as nx
from grid_loader import Grid
import numpy as np

def euclidean(node1, node2):
    x1, y1 = node1
    x2, y2 = node2
    return ((x1-x2)**2+(y1-y2)**2)**0.5

class AStar:

    # Constructor
    def __init__(self):
        self.graph = None
        self.grid_res = None    # m / pixel

    def load_grid(self, grid_obj: Grid, occ_thresh = 0.5):
        """
            Load a given Grid object into a networkx graph
            The edges are given a weight 1 and the occupied
            cells are removed
            
            Parameters:
            - grid_obj: Grid
                A Grid object that is to be loaded for path
                finding
            - occ_thresh: float (default: 0.5)
                A threshold value for depicting occupied cell
                If cell value >= occ_thresh, it is considered
                occupied and removed
            Returns:
            - removed_nodes: int
                The number of nodes that were removed from
                grid (number of occupied cells)
        """
        self.grid_res = grid_obj.grid_res   # Useful for translation from px to m and back
        self.graph = nx.grid_2d_graph(grid_obj.w, grid_obj.h)
        removed_nodes = 0
        for i in range(grid_obj.w):
            for j in range(grid_obj.h):
                if grid_obj.grid_data[i, j] >= occ_thresh: # Occupied
                    self.graph.remove_node((i, j))
                    removed_nodes += 1
        # Set edge properties of the graph
        nx.set_edge_attributes(self.graph, {e: 1 for e in self.graph.edges()}, "cost")
        return removed_nodes
    
    # Return a route of [x, y] points
    def get_route(self, start, end, heuristic = euclidean, weight = 0.5):
        start_px = tuple((np.array(start) / self.grid_res).astype(int))
        end_px = tuple((np.array(end) / self.grid_res).astype(int))
        astar_path = nx.astar_path(self.graph, start_px, end_px,
            heuristic=lambda n1, n2: weight*heuristic(n1, n2), weight="cost")
        astar_path = np.array(astar_path)
        astar_path_m = astar_path * self.grid_res
        return astar_path_m
