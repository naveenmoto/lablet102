import time
import toml
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import lidar 


config_params = toml.load("config.toml")['params']
print(config_params)
locals().update(config_params)

r =0.1 
l =0.1
circles = [(0, 0, r), (0, l, r), (0, -l, r)]

def circle_collision_check(grid, local_traj,grid_res):
    xmax, ymax = grid.shape
    all_x = np.arange(xmax)
    all_y = np.arange(ymax)
    X, Y = np.meshgrid(all_x, all_y)
    for xl, yl, tl in local_traj:
      rot = np.array([[np.sin(tl), -np.cos(tl)],[np.cos(tl), np.sin(tl)]])
      for xc, yc, rc in circles:
        xc_rot, yc_rot = rot @ np.array([xc, yc]) + np.array([xl, yl])
        xc_pix, yc_pix = int(xc_rot/grid_res), int(yc_rot/ grid_res)
        rc_pix = (rc/ grid_res)
        inside_circle = ((X-xc_pix)**2 +(Y-yc_pix)**2 - rc_pix**2 < 0)
        occupied_pt = grid[X, Y] <= 0.7
        if np.sum(np.multiply( inside_circle, occupied_pt)):
          return True
    return False

def euclidean(pt1,pt2):
  return ((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**(1/2)


def simulate_unicycle(pose, v,w, N=1, dt=0.5):
    x, y, t = pose
    poses = []
    for _ in range(N):
        x += v*np.cos(t)*dt
        y += v*np.sin(t)*dt
        t += w*dt
        poses.append([x,y,t])
    return np.array(poses)

def command_window(v, w, dt=0.1):
    """Returns acceptable v,w commands given current v,w"""
    # velocity can be (0, V_MAX)
    # ACC_MAX = max linear acceleration
    v_max = min(V_MAX, v + ACC_MAX*dt)
    v_min = max(0, v - ACC_MAX*dt)
    # omega can be (-W_MAX, W_MAX)
    #W_DOT_MAX = max angular acceleration
    epsilon = 1e-6
    w_max = min(W_MAX, w + W_DOT_MAX*dt)
    w_min = max(-W_MAX, w - W_DOT_MAX*dt)
    
    #generate quantized range for v and omega
    vs = np.linspace(v_min, v_max, num=11)
    ws = np.linspace(w_min, w_max, num=21)
    
    #cartesian product of [vs] and [ws]
    #remember there are 0 velocity entries which have to be discarded eventually
    commands = np.transpose([np.tile(vs, len(ws)), np.repeat(ws, len(vs))])
    
    #calculate kappa for the set of commands
    kappa = commands[:,1]/(commands[:,0]+epsilon)
    
    #returning only commands < max curvature 
    #return commands[(kappa < K_MAX) & (commands[:, 0] != 0)]
    return commands[commands[:, 0] != 0]


def track(ref_path, pose, v, w, grid,dt=0.1):
    commands = command_window(v, w, dt)
    #initialize path cost
    best_cost, best_command = np.inf, None
    ld=lidar.Lidar(max_dist=4) 
    ld.set_env(grid,1)
    beam_data=ld.sense_obstacles(pose)
    for i in range(len(beam_data)):
      beam_data[i]=beam_data[i]-4
    for i, (v, w) in enumerate(commands):
        local_path = simulate_unicycle(pose,v,w,N=20) #Number of steps = prediction horizon
        
    
        
        if any(beam_data):
          if circle_collision_check(grid, local_path,1): #ignore colliding paths
            continue
        
        #calculate cross-track error
        #can use a simplistic definition of 
        #how close is the last pose in locstart_pose=np.array([start[0],start[1],0])
        
        cte =euclidean(ref_path[-1],local_path[-1])
        
        #other cost functions are possible
        #can modify collision checker to give distance to closest obstacle
        #cost = w_cte*cte + w_speed*(V_MAX - v)**2 
        
        #check if there is a better path_indexcandidate
        if cte < best_cost:
            best_cost, best_command = cte,commands[i]

    if best_command[0] or best_command[1]:
        return best_command
    else:
        return [0, 0]
        
        
def exec_dwa(start_pose,astar_path,grid,interpolate):
  i=0
  ref_path=np.array(astar_path)
  ref_path2=[]
  while i < len(ref_path)-1:
    x=np.linspace(ref_path[i,0],ref_path[i+1,0],interpolate)
    y=np.linspace(ref_path[i,1],ref_path[i+1,1],interpolate)
    for j in range(interpolate):
      ref_path2.append([x[j],y[j]])
    i=i+1
  ref_path2=np.array(ref_path2)
  pose = start_pose
  logs = []
  path_index = 0
  v, w = 0.0, 0.0
  while path_index < len(ref_path2)-1:
      t0 = time.time()
      local_ref_path = ref_path2[path_index:path_index+pred_horizon]
      # update path_index using current pose and local_ref_path
      if euclidean(pose[:2],local_ref_path[-1,:2])<goal_threshold*10:
        path_index=path_index+1
      # get next command
      v, w = track(local_ref_path,pose,v,w,grid)
      
      #simulate vehicle for 1 step
      # remember the function now returns a trajectory, not a single pose
      pose = simulate_unicycle(pose,v,w,N=25)[0]
     
      
      #update logs
      logs.append([*pose, v, w])
      t1 = time.time() #simplest way to time-profile your code
      print(path_index)
  return logs

