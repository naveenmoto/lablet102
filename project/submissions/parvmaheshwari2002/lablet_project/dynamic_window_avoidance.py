import toml
import numpy as np
from collision_checking import CollisionChecking

class DWA:
    def __init__(self,config_file_dict,grid_shape,grid_res):
        self.grid_res =grid_res

        config_params = config_file_dict['params']
        self.V_MAX = config_params["V_MAX"]
        self.ACC_MAX = config_params["ACC_MAX"]
        self.W_MAX = config_params["W_MAX"]
        self.W_DOT_MAX = config_params["W_DOT_MAX"]
        self.K_MAX = config_params["K_MAX"]
        self.pred_horizon = round(config_params["pred_horizon"]/self.grid_res)
        self.goal_threshold = config_params["goal_threshold"]
        self.w_cte = config_params["w_cte"]
        self.w_speed = config_params["w_speed"]
        self.w_err = config_params["w_err"]

        self.collision_checker = CollisionChecking(grid_shape,self.grid_res)
        
    def simulate_unicycle(self,pose, v,w, N=1, dt=0.1):
        x, y, t = pose
        poses = []
        for _ in range(N):
            x += v*np.cos(t)*dt
            y += v*np.sin(t)*dt
            t += w*dt

            t = np.arctan2(np.sin(t), np.cos(t))
            poses.append([x,y,t])
        return np.array(poses)

    def command_window(self,v, w, dt=0.1):
        """Returns acceptable v,w commands given current v,w"""
        v_max = min(self.V_MAX, v + self.ACC_MAX*dt)
        v_min = max(0, v - self.ACC_MAX*dt)
        epsilon = 1e-6
        w_max = min(self.W_MAX, w + self.W_DOT_MAX*dt)
        w_min = max(-self.W_MAX, w - self.W_DOT_MAX*dt)
        
        #generate quantized range for v and omega
        vs = np.linspace(v_min, v_max, num=11)
        ws = np.linspace(w_min, w_max, num=21)
        
        #cartesian product of [vs] and [ws]
        #remember there are 0 velocity entries which will be discarded eventually
        commands = np.transpose([np.tile(vs, len(ws)), np.repeat(ws, len(vs))])
        
        #calculate kappa for the set of commands
        kappa = commands[:,1]/(commands[:,0]+epsilon)
        
        #returning only commands < max curvature 
        return commands[(kappa < self.K_MAX) & (commands[:, 0] != 0)]
        # return commands[(kappa < self.K_MAX)]

    def track(self,grid,ref_path, pose, v, w, dt=0.1):
        commands = self.command_window(v, w, dt)
        #initialize path cost
        best_cost, best_command, counter = np.inf, None, None
        for i, (v, w) in enumerate(commands):
            local_path = self.simulate_unicycle(pose,v,w,N = self.pred_horizon) #Number of steps = prediction horizon
            if self.collision_checker.circle_collision_check(grid, local_path): #ignore colliding paths
                continue
            
            #calculate cross-track error - perpendicular offset of the last point of the local path from the refernce path
            
            ref_dist = list(np.hypot((ref_path[:,0]-local_path[-1,0]),(ref_path[:,1]-local_path[-1,1])))
            ref_idx = ref_dist.index(min(ref_dist))
            cte = ref_dist[ref_idx]

            err = np.hypot((ref_path[-1,0]-local_path[-1,0]),(ref_path[-1,1]-local_path[-1,1])) #cross track error between the last point of the reference path and the last point of the local path according to current commands
            cost = self.w_cte*cte + self.w_speed*(self.V_MAX - v)**2 + self.w_err * err
            
            #check if there is a better candidate
            if cost < best_cost:
                best_cost, best_command = cost, commands[i]
                counter = 1

        if counter :
            return best_command
        else:
            return [0, 0]
