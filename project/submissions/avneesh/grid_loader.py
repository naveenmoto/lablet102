"""
    Grid loader that contains the Grid class
"""

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Grid loader and manipulator
class Grid:

    # Constructor
    def __init__(self):
        self.grid_img = None
        self.grid_data = None
        self.grid_res = None
        self.w_m = None
        self.h_m = None
        self.w = None
        self.h = None
    
    # Set grid resolution
    def set_grid_res(self, grid_res):
        """
            Sets the grid resolution and other associated variables
            Parameters:
            - grid_res: float
                Grid resolution (in m / pixel)
        """
        self.grid_res = grid_res
        # Extract other data
        self.w, self.h = self.grid_img.size # in pixels
        self.w_m, self.h_m = np.array([self.w, self.h]) * self.grid_res # in m
        pass

    # Load grid from image
    def load_from_image(self, img_path, grid_res = 0.05):
        """
            Load image from disk at a given grid resolution
            Parameters:
            - img_path: str
                The path to image (on disk). Must be a `.png`
            - grid_res: float (default = 0.05)
                The grid resolution (in m/pixel)
        """
        
        # Load PIL image
        # 1. Read from path
        # 2. Convert to grayscale (0 to 255) -> (white to black)
        # 3. Flip top and bottom for origin to be at bottom left (instead of top left)
        self.grid_img = Image.open(img_path).convert('L').transpose(Image.FLIP_TOP_BOTTOM)
        self.set_grid_res(grid_res)
        # Convert to numpy grid
        # img: 0 (black / occupied) to 255 (white / free)
        # grid_data / occupancy grid: 0 (free) to 1 (occupied): Probabilistic
        # img: first axis is Y, second is X, grid: first X then Y
        self.grid_data = 1 - np.array(self.grid_img).T / 255

    # Show image
    def show_grid(self):
        plt.figure()
        plt.imshow(self.grid_data.T, cmap=plt.cm.gray_r, origin='lower', 
                    extent=[0, self.w_m, 0, self.h_m])
    
    # Add obstacles
    def add_obstacles(self, obstacles):
        """
            Add an array of obstacles
            Parameters:
            - obstacles: list
                A list of [x1, y1, x2, y2] values (all in m) depicting
                the obstacles to be added
        """
        for obstacle in obstacles:
            x1, y1, x2, y2 = (np.array(obstacle) / self.grid_res).astype(int)
            self.grid_data[x1:x2, y1:y2] = 1.0
