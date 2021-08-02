import numpy as np

class CollisionChecking:

    def __init__(self,grid_size,grid_res):
        self.grid_res = grid_res
        self.grid_size = grid_size
        self.w=1
        self.l=2
        self.r = 2.0
        self.offset = 0.1
        self.circles = [(0, 0, self.r), (0, self.offset, self.r), (0,-self.offset, self.r)]
    
    def circle_collision_check(self, grid, local_traj):
        xmax, ymax = grid.shape
        all_x = np.arange(xmax)
        all_y = np.arange(ymax)
        X, Y = np.meshgrid(all_x, all_y)
        for xl, yl, tl in local_traj:
            rot = np.array([[np.sin(tl), -np.cos(tl)],[np.cos(tl), np.sin(tl)]])
            for xc, yc, rc in self.circles:
                xc_rot, yc_rot = rot @ np.array([xc, yc]) + np.array([xl, yl])
                xc_pix, yc_pix = int(xc_rot/self.grid_res), int(yc_rot/ self.grid_res)
                rc_pix = (rc/ self.grid_res)
                inside_circle = ((X-xc_pix)**2 +(Y-yc_pix)**2 - rc_pix**2 < 0)
                occupied_pt = grid[X, Y] != 0
                if np.sum(np.multiply( inside_circle, occupied_pt)):
                    return True
        return False
