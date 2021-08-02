from lidar_simulator import Lidar
from global_planner import Planner
from dynamic_window_avoidance import DWA
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import toml

config_path="../config/config.toml"
config_file_dict = toml.load(config_path)
grid = np.zeros([40, 40]) #initialising the environment
grid[8:12,6:14] = 1.0 #known obstacle
grid_res = 1.0

og_grid = copy.deepcopy(grid) #og_grid is the envrionment visible to and updated by the vehicle

start = eval(config_file_dict['positions']['start']) #p
goal = eval(config_file_dict['positions']['goal']) #p

start_node = (int(start[0]/grid_res), int(start[1]/grid_res))
goal_node = (int(goal[0]/grid_res), int(goal[1]/grid_res))

planner = Planner(grid)

#bounds for unknow obstacles
lower_x_obst = 15
upper_x_obst = 25
lower_y_obst = 15
upper_y_obst = 25
lower_x_obst,upper_x_obst,lower_y_obst,upper_y_obst = int(lower_x_obst/grid_res),int(upper_x_obst/grid_res),int(lower_y_obst/grid_res),int(upper_y_obst/grid_res) 
grid[lower_x_obst:upper_x_obst,lower_y_obst:upper_y_obst] = 0.4

#x1,x2,y1,y2 - extent for plotting the graph using imshow
x1, y1 = 0, 0
x2, y2 = grid.shape[0] * grid_res , grid.shape[1] * grid_res

fig, ax = plt.subplots(figsize=(12,12))
plt.title("Grid with Pre Defined Obstacles.")
plt.figtext(0.40,0.15,"Pre Defined Obstacles marked in grey and Free space marked in green")
ax.imshow(og_grid.T,origin = 'lower', cmap=plt.cm.Dark2,extent = [x1,x2,y1,y2])
ax.set_xlabel('All measurements are in meters')
ax.set_ylabel('All measurements are in meters')
plt.show()

x,y = planner.compute_path(start_node,goal_node)
x,y = np.asarray(x).reshape((-1,1)) * grid_res ,np.asarray(y).reshape((-1,1)) *grid_res
path = np.concatenate((x,y),axis=1)

fig, ax = plt.subplots(figsize=(12,12))
plt.title("Grid with all the obstacles.")
plt.figtext(0.40,0.15,"Predefined Obstacles marked in grey and, \nObstacles in environment unknown to the vehicle are marked in pink")
ax.imshow(grid.T,origin = 'lower', cmap=plt.cm.Dark2,extent = [x1,x2,y1,y2])
ax.scatter(start[0],start[1], marker = "+", color = "yellow", s = 200, )
ax.scatter(goal[0],goal[1], marker = "+", color = "red", s = 200)
ax.scatter(x,y, marker = ".", color = "blue", s = 50)
ax.legend(['True Start Point','Goal Point', 'A star path - Observe that this passes through unobserved obstacles'])
ax.set_xlabel('All measurements are in meters')
ax.set_ylabel('All measurements are in meters')
plt.show()
start_cord_x = min(x2,max(x1,start[0]+np.random.normal(0,1)))
start_cord_y = min(x2,max(x1,start[1]+np.random.normal(0,1)))
start_pose = (start_cord_x,start_cord_y,np.random.normal(0,np.pi/2)) # position includes some error from true start point and vehicle has random initial heading
pose = start_pose
logs = []
path_index = 0
v, w = 0.0, 0.0

dwa=DWA(config_file_dict,grid.shape,grid_res)
lidar = Lidar(max_dist=10.0)
lidar.set_env(grid,grid_res)
while path_index < len(path)-1:
    t0 = time.time()
    local_path = path[path_index:min(len(path)-1,path_index+dwa.pred_horizon)]
    print(pose)
    # update path_index using current pose and local_path
    ref_dist = list(np.hypot((path[:,0]-pose[0]),(path[:,1]-pose[1])))
    path_index = ref_dist.index(min(ref_dist))

    dist = lidar.sense_obstacles(pose=pose)
    for reading, tprime in zip(dist, lidar.beam_angles):
        x_obst, y_obst , b = reading
        if x_obst != -1:
            og_grid[x_obst,y_obst] = 0.2

    # get next command
    v, w = dwa.track(og_grid,local_path,pose,v,w)
    
    #simulate vehicle for 1 step
    pose = dwa.simulate_unicycle(pose,v,w)[0]
    
    #update logs
    logs.append([*pose, v, w])
    t1 = time.time()
    print(f"idx:{path_index}, v:{v:0.3f}, w:{w:0.3f}, time:{(t1-t0) * 1000:0.1f}ms")

    if np.hypot((path[-1,0]-pose[0]),(path[-1,1]-pose[1])) < dwa.goal_threshold:
        break

poses = np.array(logs)[:,:3]

fig, ax = plt.subplots(figsize=(12,12))
plt.title("Final Result - All measurements are in meters")
plt.figtext(0.45,0.15,"Predefined Obstacles marked in grey and \nDetected obstacles in environment are marked in orange")
ax.imshow(og_grid.T,origin = 'lower', cmap=plt.cm.Dark2,extent = [x1,x2,y1,y2])
plt.quiver(start_pose[0],start_pose[1], np.cos(start_pose[2]), np.sin(start_pose[2]),scale=12)
# ax.scatter(start_pose[0],start_pose[1], marker = "+", color = "yellow", s = 200)
ax.scatter(goal[0],goal[1], marker = "+", color = "red", s = 200)
ax.scatter(path[:,0],path[:,1], marker = ".", color = "blue", s = 50)
ax.scatter(poses[:,0],poses[:,1], marker = ".", color = "yellow", s = 50)
ax.legend(['Real Start Point and orientation','Goal Point', 'A star path','DWA or Actual path traversed'])
ax.set_xlabel('All measurements are in meters')
ax.set_ylabel('All measurements are in meters')
plt.show()
