import numpy as np

"""Lidar sensor"""

class Lidar(object):
    def __init__(self, nbeams=7, fov=60, max_dist=5.0):
        '''
        nbeams = number of sensing beams
        FOV = field-of-view of Lidar/ coverage in degrees
        max_dist = maximum distance Lidar can sense
        sampling_pts = num pts on a given beam for obstacle check
        '''
        self.beam_angles = np.deg2rad(np.linspace(-fov/2+np.pi/2,fov/2+np.pi/2,num=nbeams))
        self.lidar_range = max_dist
        self.sampling_pts = 20
    
    def set_env(self, grid, grid_res=0.05):
        #2-D occupancy grid and grid_resolution
        self.grid = grid
        self.grid_res = grid_res
    
    def sense_obstacles(self, pose):
        xc, yc, theta = pose
        line_sampler = self.lidar_range * np.linspace(0, 1, num=self.sampling_pts)
        beam_data = []
        for idx, b in enumerate(self.beam_angles):
            direction = np.array([np.cos(theta+b), np.sin(theta+b)])
            beam_data.append((-1,-1,10))
            for d in line_sampler:
                beam_x, beam_y = np.array([xc, yc]) + d * direction
                i, j = int(beam_x/self.grid_res), int(beam_y/self.grid_res) #i,j gives the row and columnof the detected obstalce in the grid array
                if i< self.grid.shape[0] and j< self.grid.shape[1] and self.grid[i][j] != 0:
                    beam_data[idx]=(i,j,d)
                    break
        return beam_data
