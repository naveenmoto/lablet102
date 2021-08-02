import numpy as np
from lidar import Lidar
import toml
######################################################
config_params = toml.load('config.toml')['params']
locals().update(config_params)
######################################################


def simulate_unicycle(pose, v,w, N=1, dt=0.1):
    x = pose[0] + v*np.cos(pose[2])*dt
    y = pose[1] + v*np.sin(pose[2])*dt
    theta = pose[2] + w*dt
    all_poses = [[x,y,theta]]
    for _ in range(N-1):
        xi = all_poses[-1][0] + v*np.cos(all_poses[-1][2])*dt
        yi = all_poses[-1][1] + v*np.sin(all_poses[-1][2])*dt
        thetai = all_poses[-1][2] + w*dt
        all_poses.append([xi,yi,thetai])
    return all_poses

def command_window(v, w, dt=0.1):
    """Returns acceptable v,w commands given current v,w"""
    v_max = min((v + ACC_MAX*dt),V_MAX)
    v_min = max((v - ACC_MAX*dt),0)
    
    w_max = min((w + W_DOT_MAX*dt),W_MAX)
    w_min = max((w - W_DOT_MAX*dt),-W_MAX)
    
   
    vs = np.linspace(v_min, v_max, num=5) 
    ws = np.linspace(w_min, w_max, num=10)
    
    commands = np.transpose([np.tile(vs, len(ws)), np.repeat(ws, len(vs))])
    epsilon = 1e-6
    
    kappa = commands[:,1]/(commands[:,0] + epsilon)
    
    return commands[(kappa < K_MAX) & (commands[:, 0] != 0)]
  
def track(local_ref_path, pose, v, w, Lid, dt=0.1):
    commands = command_window(v, w)
    best_cost, best_command = np.inf, None
    for i, [v, w] in enumerate(commands):
        local_path = simulate_unicycle(pose,v,w,pred_horizon) #Number of steps = prediction horizon
        
        ref_x = local_ref_path[-1][0]
        ref_y = local_ref_path[-1][1]
        ref_t = local_ref_path[-1][2]
        [xc,yc,t] = local_path[-1]

        cte = (ref_x-xc)**2+ (ref_y-yc)**2
        
        cost = w_cte*cte  + w_speed*(V_MAX - v)**2 + w_col*(0 if Lid == None else Lid.lidar_range-min(Lid.sense_obstacles((ref_x,ref_y,ref_t))))
        if cost < best_cost:
            best_cost, best_command = cost, [v,w]

    if best_command != None:
        return best_command
    else:
        return [0, 0]


def DWA_tracker(start_pose,ref_path,grid):  
    lidar = Lidar(max_dist=5)  
    lidar.set_env(grid,grid_res=1)        
    pose = start_pose
    logs = []
    path_index = 0
    v, w = 0.0, 0.0
    while path_index < len(ref_path)-1:
        local_ref_path = ref_path[path_index:path_index+pred_horizon]
        x,y= pose[0][0],pose[0][1]
        ref_x,ref_y  = local_ref_path[-1][0],local_ref_path[-1][1]
        if np.sqrt((ref_x-x)**2+(ref_y-y)**2)>=goal_threshold:
            path_index += 1
        [v, w] = track(local_ref_path,pose[0],v,w,lidar)
        pose = simulate_unicycle(pose[0],v,w,pred_horizon,dt)
        logs.append([*(pose[0]), v, w])    
        poses = np.array(logs)[:,:3]
    return poses
