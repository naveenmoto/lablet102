import numpy as np
import heapq
from collections import namedtuple

DEBUG = False
SQRT2 = np.sqrt(2)

def _euclidean(node1, node2):
    x1, y1 = node1
    x2, y2 = node2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


Node = namedtuple("Node", ["x", "y"])


class Astar:
    def __init__(self, occupancy_grid):
        """Creates a wrapper over an occupancy grid on which astar algorithm is applied.

        Args:
            occupancy_grid (2d numpy array): A 2d grid with 1 indicates object, 0 indicates free space.
        """
        self.occupancy_grid = occupancy_grid

    def shortest_path(self, start, goal, norm=_euclidean):

        # Convert start and goal to namedtuples for better readability
        start = Node(*start)
        goal = Node(*goal)

        # Following heap is used to get the next node to process in constant time.
        heap = []

        self.finalized = np.zeros_like(self.occupancy_grid)

        # predecessors is a 3d tensor, that can be viewed as a 2d lookup table where the entry is the index of( in (y, x) form) previous node
        # in the shortest path from starting node to current node.
        self.predecessors = -np.ones(
            (self.occupancy_grid.shape[0],
             self.occupancy_grid.shape[1], 2))

        # Each entry corresponds to the best distance so far from the shortest node to current node
        # Initialized to infinity
        self.distance = np.inf * np.ones_like(self.occupancy_grid)

        heapq.heappush(heap, (0+norm(start, goal), start))
        self.distance[start.y, start.x] = 0

        while len(heap):
            # Heap pop will return (distance, node).
            # But we need only the distance
            _, node = heapq.heappop(heap)

            # If the shortest distance to current node is finalized,
            # we need not process that node
            if self.finalized[node.y, node.x] == 1:
                continue

            # We can stop the algorithm ones we reach the goal
            if node == goal:
                break

            # Set current node as finalized
            self.finalized[node.y, node.x] = 1

            YMAX = self.occupancy_grid.shape[0]
            XMAX = self.occupancy_grid.shape[1]

            if node.y < YMAX-1:
                if not self.finalized[node.y+1, node.x] \
                        and not self.occupancy_grid[node.y+1, node.x]:
                    if self.distance[node.y+1, node.x] \
                            > 1+self.distance[node.y, node.x]:
                        self.predecessors[node.y+1, node.x] = node

                        heapq.heappush(heap, (self.distance[node.y, node.x]+1
                                              + norm((node.x, node.y+1), goal), Node(node.x, node.y+1)))
                    self.distance[node.y+1, node.x] = min(
                        self.distance[node.y+1, node.x], 1+self.distance[node.y, node.x])

            if node.y > 0:
                if not self.finalized[node.y-1, node.x] and not self.occupancy_grid[node.y-1, node.x]:
                    if self.distance[node.y-1, node.x] > 1+self.distance[node.y, node.x]:
                        self.predecessors[node.y-1, node.x] = node
                        heapq.heappush(heap, (self.distance[node.y, node.x]+1
                                              + norm((node.x, node.y-1), goal), Node(node.x, node.y-1)))
                    self.distance[node.y-1, node.x] = min(
                        self.distance[node.y-1, node.x], 1+self.distance[node.y, node.x])

            if node.x < XMAX-1:
                if not self.finalized[node.y, node.x+1] and not self.occupancy_grid[node.y, node.x+1]:
                    if self.distance[node.y, node.x+1] > 1+self.distance[node.y, node.x]:
                        self.predecessors[node.y, node.x+1] = node
                        heapq.heappush(heap, (self.distance[node.y, node.x]+1
                                              + norm((node.x+1, node.y), goal), Node(node.x+1, node.y)))
                    self.distance[node.y, node.x+1] = min(
                        self.distance[node.y, node.x+1], 1+self.distance[node.y, node.x])

            if node.x > 0:
                if not self.finalized[node.y, node.x-1] and not self.occupancy_grid[node.y, node.x-1]:
                    if self.distance[node.y, node.x-1] > 1+self.distance[node.y, node.x]:
                        self.predecessors[node.y, node.x-1] = node
                        heapq.heappush(heap, (self.distance[node.y, node.x]+1
                                              + norm((node.x-1, node.y), goal), Node(node.x-1, node.y)))
                    self.distance[node.y, node.x-1] = min(
                        self.distance[node.y, node.x-1], 1+self.distance[node.y, node.x])

            if node.y < YMAX-1 and node.x < XMAX-1:
                if not self.finalized[node.y+1, node.x+1] \
                        and not self.occupancy_grid[node.y+1, node.x+1]:
                    if self.distance[node.y+1, node.x+1] \
                            > SQRT2+self.distance[node.y, node.x]:
                        self.predecessors[node.y+1, node.x+1] = node

                        heapq.heappush(heap, (self.distance[node.y, node.x]+SQRT2
                                              + norm((node.x+1, node.y+1), goal), Node(node.x+1, node.y+1)))
                    self.distance[node.y+1, node.x+1] = min(
                        self.distance[node.y+1, node.x+1], SQRT2+self.distance[node.y, node.x])

            if node.y > 0 and node.x < XMAX-1:
                if not self.finalized[node.y-1, node.x+1] and not self.occupancy_grid[node.y-1, node.x+1]:
                    if self.distance[node.y-1, node.x+1] > 1+self.distance[node.y, node.x]:
                        self.predecessors[node.y-1, node.x+1] = node
                        heapq.heappush(heap, (self.distance[node.y, node.x]+SQRT2
                                              + norm((node.x+1, node.y-1), goal), Node(node.x+1, node.y-1)))
                    self.distance[node.y-1, node.x+1] = min(
                        self.distance[node.y-1, node.x+1], SQRT2+self.distance[node.y, node.x])

            if node.x > 0 and node.y > 0:
                if not self.finalized[node.y-1, node.x-1] and not self.occupancy_grid[node.y-1, node.x-1]:
                    if self.distance[node.y-1, node.x-1] > 1+self.distance[node.y, node.x]:
                        self.predecessors[node.y-1, node.x-1] = node
                        heapq.heappush(heap, (self.distance[node.y, node.x]+SQRT2
                                              + norm((node.x-1, node.y-1), goal), Node(node.x-1, node.y-1)))
                    self.distance[node.y-1, node.x-1] = min(
                        self.distance[node.y-1, node.x-1], SQRT2+self.distance[node.y, node.x])

            if node.x > 0 and node.y < YMAX - 1:
                if not self.finalized[node.y+1, node.x-1] and not self.occupancy_grid[node.y+1, node.x-1]:
                    if self.distance[node.y+1, node.x-1] > 1+self.distance[node.y, node.x]:
                        self.predecessors[node.y+1, node.x-1] = node
                        heapq.heappush(heap, (self.distance[node.y, node.x]+SQRT2
                                              + norm((node.x-1, node.y+1), goal), Node(node.x-1, node.y+1)))
                    self.distance[node.y+1, node.x-1] = min(
                        self.distance[node.y+1, node.x-1], SQRT2+self.distance[node.y, node.x])

        path = []

        node = goal
        while node.x != -1:
            path.append(node)
            node = Node(*self.predecessors[int(node.y), int(node.x)])

        path.reverse()

        path = np.array(path)
        x, y = path[:, 0], path[:, 1]
        if DEBUG:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.distance, origin="lower")
            plt.title("Explored Area")
            plt.subplot(122)
            plt.imshow(1-self.occupancy_grid, origin="lower", cmap="gray")
            plt.title("A* Path")
            plt.plot(x, y)
            plt.show()
        return path[:, 0], path[:, 1]
