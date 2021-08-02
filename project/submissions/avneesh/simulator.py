"""
    Simulators for everything
    1. LiDAR
    2. Unicycle model
"""

import numpy as np
from grid_loader import Grid

class Lidar:
    # Constructor
    def __init__(self, fov_deg=60.0, nbeams=7, max_dist=5.0, sampling_pts=20):
        """
            Returns a Lidar (Simulated) object

            Paraemters:
            - fov_deg: float
                Field of view (range of angles). The coverage
                of the Lidar
            - nbeams: int
                Number of beams for covering the field of view
            - max_dist: float
                Maximum sensing distance for each beam
            - sampleing_pts: int
                Number of points to sample per scanning line
        """
        min_angle = np.deg2rad(-fov_deg/2)
        max_angle = np.deg2rad(fov_deg/2)
        self.scan_angles = np.linspace(min_angle, max_angle, nbeams)
        self.scan_line_samples = np.linspace(0, max_dist, sampling_pts)
        # Grid properties
        self.grid = None
        self.grid_res = None

    # Set the grid for the LiDAR
    def set_grid(self, grid: Grid):
        """
            Set the sensing grid (simulated world) for the LiDAR

            Parameters:
            - grid: Grid
                The Grid class object that contains the grid_data
                at a specified grid resolution
        """
        self.grid = grid.grid_data  # [x, y], numpy array
        self.grid_res = grid.grid_res   # Resolution (m / pixel)
    
    # Beam data response
    def get_beam_data(self, pose):
        """
            Get the distance readings on each beam / scan line

            Paremeters:
            - pose: tuple
                Values (x, y, theta) for the robot pose (theta
                in radians)
            
            Returns:
            - bran_data: numpy array
        """
        xc, yc, thetac = pose
        beam_data = []
        for sc_ang in self.scan_angles:
            scan_dir = np.array([np.cos(thetac+sc_ang), np.sin(thetac+sc_ang)])
            for sc_dist in self.scan_line_samples:
                beam_x, beam_y = np.array([xc, yc]) + sc_dist * scan_dir
                xp, yp = int(beam_x/self.grid_res), int(beam_y/self.grid_res)
                if self.grid[xp, yp] >= 0.5:
                    break
            beam_data.append(sc_dist)
        return beam_data

# Unicycle simulation
def simulate_unicycle(pose, v, w, N=1, dt=0.1):
    x, y, t = pose
    poses = []
    for _ in range(N):
        x += v*np.cos(t)*dt
        y += v*np.sin(t)*dt
        t += w*dt
        poses.append([x, y, t])
    return np.array(poses)
