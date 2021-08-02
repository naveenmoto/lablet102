from simulator import simulate_unicycle
from astar_path import euclidean
# Modules
import numpy as np
import time


class DWA:
    def __init__(self, v_max, w_max, acc_max, w_dot_max, k_max):
        self.V_MAX = v_max
        self.W_MAX = w_max
        self.ACC_MAX = acc_max
        self.W_DOT_MAX = w_dot_max
        self.K_MAX = k_max
        self.pred_horizon = 10
        self.goal_threshold = 0.05
        self.pose_cost = lambda pose: 0.0   # Cost of being at 'pose'
    
    def set_path_props(self, pred_horizon, goal_threshold):
        self.pred_horizon = pred_horizon
        self.goal_threshold = goal_threshold

    def set_weights(self, w_cte, w_speed, w_kappa, w_position):
        self.w_cte = w_cte
        self.w_speed = w_speed
        self.w_kappa = w_kappa
        self.w_position = w_position

    def command_window(self, v, w, dt=0.1):
        """
            Returns acceptable v, w commands given current v, w
        """
        # Max and min velocity (0, V_MAX)
        v_max = min(self.V_MAX, v + self.ACC_MAX*dt)
        v_min = max(0, v - self.ACC_MAX*dt)
        # Max and min angular velocity (-W_MAX, W_MAX)
        w_max = min(self.W_MAX, w + self.W_DOT_MAX*dt)
        w_min = max(-self.W_MAX, w - self.W_DOT_MAX*dt)
        # Generate quantized range
        v_vals = np.linspace(v_min, v_max, num=11)
        w_vals = np.linspace(w_min, w_max, num=21)
        # Get commands as a cartesian product of [v_vals] and [w_vals]
        commands = np.transpose([np.tile(v_vals, len(w_vals)), np.repeat(w_vals, len(v_vals))])
        # Kappa values (1/R) = w / v
        epsilon = 1e-6  # To prevent divide by 0
        kappa_vals = np.abs(commands[:, 1])/(commands[:, 0] + epsilon)
        # Return only the commands that have a stable curvature and non-zero velocity
        return commands[(commands[:, 0] != 0) & (kappa_vals < self.K_MAX)]
    
    def best_control_option(self, ref_path, pose, v, w, dt=0.1):
        """
            Get the best control 'v' and 'w' values given the
            parameters

            Paraemters
            - ref_path: list or np.ndarray
                A list of [x, y, th] points (or [x, y] points).
                Reference path, must be a local segment of the
                global path
            - pose: list
                Current pose of the robot as [x, y, theta]
            - v: float
                Current velocity
            - w: float
                Current angular velocity
        """
        commands = self.command_window(v, w, dt) # Get the list of [v, w]
        # Best command
        best_command_cost, best_command = np.inf, None
        for i, (cv, cw) in enumerate(commands):
            # Local path for the particular command
            local_path = simulate_unicycle(pose, cv, cw, N=self.pred_horizon, dt=dt)
            # Cross track error
            #  find the closest point (in ref path) from last local point
            closest_point_index = np.argmin(np.hypot(ref_path[:][:2], local_path[-1, :2]))
            #  main cte calculation
            cte = euclidean(local_path[-1, :2], ref_path[closest_point_index][:2])
            # Cost of being at 'pose'
            position_cost = self.pose_cost(pose)
            # Cost of the path (consider different things)
            cost = self.w_cte * cte + self.w_speed * (self.V_MAX - v) ** 2 \
                + self.w_position * position_cost + self.w_kappa * (abs(cw)/(abs(cv)+1e-6))
            # Check if this is a better cost
            if cost < best_command_cost:
                best_command_cost, best_command = cost, (cv, cw)
        # If we have a best_command
        if best_command:
            return best_command
        else:
            # Stay stationary
            return [0, 0]

    def run_dwa(self, start_pose, ref_path, t_sim_end = 30):
        """
            Run the DWA algorithm

            Parameters
            - start_pose: list
                A starting pose (x, y, theta)
            - ref_path: list or np.ndarray
                Reference path, (x, y) list or (x, y, theta) list
            - t_sim_end: float  (default: 30)
                Ending time for simulation (a fail safe)
            
            Returns
            - logs: list
                A list of -> *pose (= x, y, th), v, w, t, dt
        """
        pose = start_pose
        logs = []   # *pose (= x, y, th), v, w, t, dt
        v, w = 0.0, 0.0
        path_index = 0
        t_start = time.time()
        time_simulation = 0.0
        # Main DWA
        while path_index < len(ref_path) - 1 and time_simulation < t_sim_end:
            t0 = time.time()
            # Main work
            local_ref_path = ref_path[path_index:path_index+self.pred_horizon]
            path_index = path_index + 1 \
                if euclidean(pose[:2], local_ref_path[-1][:2]) < self.goal_threshold * 10 \
                else path_index
            v, w = self.best_control_option(local_ref_path, pose, v, w)
            pose = simulate_unicycle(pose, v, w)[-1]    # Get last element (only one element / step)
            t1 = time.time()
            # Append log
            logs.append([*pose, v, w, time_simulation, t1-t0])
            # print(f"idx: {path_index}, v: {v:.3f}, w: {w:.3f}, dt: {(t1-t0)*1000:.1f} ms")
            if v == 0 and w == 0:
                print("No good path found, command = 0")
                break
            time_simulation = time.time() - t_start
        return np.array(logs)
