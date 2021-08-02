
from types import SimpleNamespace
import numpy as np
import toml
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import time

from unicycle import simulate_unicycle
from collision_check import circle_collision_check
from lidar import Lidar

config = toml.load("config.toml")
config_params = config['params']
params = SimpleNamespace(**config_params)


grid_data = None
grid_res = 1


class DWA:
    """Implementation of DWA

    After initializing with grid_data, this class can be used as an iterator to get the simulation progress.
    """
    def __init__(self, grid_data, ref_path, start_pose, goal_threshold=0.3, grid_res=1, reality=None) -> None:
        self.grid_data = grid_data
        if reality is None:
            self.reality = grid_data.copy()
        else:
            self.reality = reality # Dynamic data
        self.ref_path = ref_path
        self.start_pose = start_pose
        self.goal_threshold = goal_threshold
        self.grid_res = grid_res
        self.path_index = 0
        self.pose = start_pose
        self.v, self.w = 0.0, 0.0
        self.failed_attempts = -1
        self.logs = []
        self.path_index = 0
        self.lidar = Lidar(max_dist=params.lidar_dist)
        self.lidar.set_env(self.reality, self.grid_res)

    def _command_window(self, v, w, dt=0.1):
        """Returns acceptable v,w commands given current v,w"""
        # velocity can be (0, V_MAX)
        # ACC_MAX = max linear acceleration
        v_max = min(params.V_MAX, v + params.ACC_MAX*dt)
        v_min = max(0, v - params.ACC_MAX*dt)
        # omega can be (-W_MAX, W_MAX)
        # W_DOT_MAX = max angular acceleration
        epsilon = 1e-6
        w_max = min(params.W_MAX, w + params.W_DOT_MAX*dt)
        w_min = max(-params.W_MAX, w - params.W_DOT_MAX*dt)

        # generate quantized range for v and omega
        vs = np.linspace(v_min, v_max, num=11)
        ws = np.linspace(w_min, w_max, num=11)

        # cartesian product of [vs] and [ws]
        # remember there are 0 velocity entries which have to be discarded eventually
        commands = np.transpose([np.tile(vs, len(ws)), np.repeat(ws, len(vs))])

        # calculate kappa for the set of commands
        kappa = commands[:, 1]/(commands[:, 0]+epsilon)

        # returning only commands < max curvature
        return commands[(kappa < params.K_MAX) & (commands[:, 0] != 0)]

    def _track(self, ref_path, pose, v, w, dt=0.1, grid_data=grid_data,
               detect_collision=True, grid_res=grid_res):
        commands = self._command_window(v, w, dt)
        # initialize path cost
        best_cost, best_command = np.inf, None
        best_local_path = None
        for i, (v, w) in enumerate(commands):
            # Number of steps = prediction horizon
            local_path = simulate_unicycle(pose, v, w, params.pred_horizon, dt)

            if detect_collision:
                # ignore colliding paths
                hit, distance = circle_collision_check(
                    grid_data, local_path, grid_res=grid_res)
                if hit:
                    print("local path has a collision")
                    continue
            else:
                distance = np.inf
            # calculate cross-track error
            # can use a simplistic definition of
            # how close is the last pose in local path from the ref path

            cte = np.linalg.norm(
                ref_path[:, 0:2]-local_path[:len(ref_path), 0:2], axis=-1)
            cte = cte * np.linspace(0,1,len(ref_path))
            cte = np.sum(cte)
            # print(cte)

            # other cost functions are possible
            # can modify collision checker to give distance to closest obstacle
            cost = params.w_cte*cte + params.w_speed*(params.V_MAX - v)**2 + params.w_obs / distance

            # check if there is a better candidate
            if cost < best_cost:
                best_cost, best_command = cost, [v, w]
                best_local_path = local_path

        if best_command:
            return best_command, best_local_path
        else:
            return [0, 0], best_local_path

    def __iter__(self):
        self.path_index = 0
        self.logs = []
        return self

    def reset(self):
        self.path_index = 0
        self.logs = []
        return self

    def __next__(self):

        if self.path_index > len(self.ref_path)-1:
            raise StopIteration
        local_ref_path = self.ref_path[self.path_index:self.path_index+params.pred_horizon]

        if self.goal_threshold > np.min(np.hypot(local_ref_path[:, 0]-self.pose[0],
                                                 local_ref_path[:, 1]-self.pose[1])):
            self.path_index = self.path_index + 1

        self.failed_attempts += 1
        if self.failed_attempts > 1600:
            self.path_index += 1
            self.failed_attempts = -1
        # get next command
        (self.v, self.w), best_local_path = self._track(local_ref_path, self.pose, self.v, self.w, dt=params.dt,
                                                        detect_collision=True, grid_data=self.grid_data)

        # simulate vehicle for 1 step
        # remember the function now returns a trajectory, not a single pose
        self.pose = simulate_unicycle(self.pose, self.v, self.w, N=1, dt=params.dt)[0]

        self.lidar.set_env(self.reality, self.grid_res)
        distances, collision_points = self.lidar.sense_obstacles(self.pose)
        # Add obstacles to grid data
        for point in collision_points:
            if point[0] != -1:
                i, j = point
                self.grid_data[i, j] = 1

        # update logs
        self.logs.append([*self.pose, self.v, self.w, self.path_index])
        print(self.path_index)
        return np.array(self.logs), distances, best_local_path
