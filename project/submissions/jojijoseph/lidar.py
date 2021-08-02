import numpy as np

class Lidar(object):
    def __init__(self, nbeams=7, fov=60, max_dist=5.0, sampling_pts=20):
        #nbeams = number of sensing beams
        #FOV = field-of-view of Lidar/ coverage in degrees
        #max_dist = maximum distance Lidar can sense
        #sampling_pts = num pts on a given beam for obstacle check
        self.nbeams = nbeams
        self.max_dist = max_dist
        self.beam_angles = np.deg2rad(np.linspace(-fov/2+np.pi/2,fov/2+np.pi/2,num=nbeams))
        self.line_sampler = max_dist * np.linspace(0, 1, num=sampling_pts)
    
    def set_env(self, grid, grid_res=0.05):
        #2-D occupancy grid and grid_resolution
        self.grid = grid
        self.grid_res = grid_res
    
    def sense_obstacles(self, pose):
        xc, yc, theta = pose
        beam_data = [self.max_dist]*self.nbeams
        collision_points = [(-1,-1)]*self.nbeams
        for idx, b in enumerate(self.beam_angles):
            direction = np.array([np.cos(theta+b), np.sin(theta+b)])
            for d in self.line_sampler:
                beam_x, beam_y = np.array([xc, yc]) + d * direction
                i, j = round(beam_x/self.grid_res), round(beam_y/self.grid_res)
                try:
                    if self.grid[i][j] == 1:
                        collision_points[idx] = i, j
                        break
                except IndexError:
                    print("Ignoring index error in lidar sense obstacles.")
            beam_data[idx] = d
        return beam_data, collision_points