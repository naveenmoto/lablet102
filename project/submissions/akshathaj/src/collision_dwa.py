import numpy as np
import matplotlib.pyplot as plt
import toml
from operator import itemgetter

params = toml.load("config.toml")['params']

def collision_params(w,l):
	mule_extents = np.array([[-w/2,-l/2],
							 [w/2, -l/2],
							 [w/2, l/2],
							 [-w/2, l/2],
							 [-w/2,-l/2]])
	r = np.hypot(l/4,w/2)
	d = l/3
	circles = [(0, 0, r), (0, d, r), (0, -d, r)]
	return mule_extents,circles
	
def plot_mule_outline(mule_extents,circles):
	plt.figure()
	#plot rectangle or just the 4 vertices
	for vertex in mule_extents:
		p, q = vertex
		plt.plot(p, q, 'b.')
	for v1, v2 in zip(mule_extents[:-1],mule_extents[1:]):
		p1, q1 = v1
		p2, q2 = v2
		plt.plot((p1, p2), (q1,q2), 'k-')
	   
	ax = plt.gca()
	for x,y,rad in circles:
		ax.add_patch(plt.Circle((x,y), rad, fill=False))
		
def simulate_unicycle(pose, v,w, N=1, dt=0.1):
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
	V_MAX,ACC_MAX,W_DOT_MAX,W_MAX,K_MAX = itemgetter('V_MAX','ACC_MAX','W_DOT_MAX','W_MAX','K_MAX')(params)
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
	return commands[(np.abs(kappa) < K_MAX*4) & (commands[:, 0] != 0)]
	# return commands[(commands[:, 0] != 0)]

def euclidean(node1,node2):
	x1,y1 = node1
	x2,y2 = node2
	return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def circle_collision_check(grid, local_traj):
	w,l,grid_res = itemgetter('width_veh','length_veh','grid_res')(params)
	_,circles = collision_params(w,l)
	grid = np.flipud(grid).T
	veh_size_max = np.hypot(w,l)
	xmax, ymax = int(max(local_traj[:,0]+2*veh_size_max)/grid_res), int(max(local_traj[:,1]+2*veh_size_max)/grid_res)
	xmin, ymin = int(min(local_traj[:,0]-2*veh_size_max)/grid_res), int(min(local_traj[:,1]-2*veh_size_max)/grid_res)
	all_x = np.arange(max(0,xmin),min(xmax,grid.shape[0]))
	all_y = np.arange(max(0,ymin),min(ymax,grid.shape[1]))
	X, Y = np.meshgrid(all_x, all_y)
	for xl, yl, tl in local_traj:
		# rotate by -(90-theta)
		rot = np.array([[np.sin(tl), np.cos(tl)],[-np.cos(tl), np.sin(tl)]])
		for xc, yc, rc in circles:
			xc_rot, yc_rot = rot @ np.array([xc, yc]) + np.array([xl, yl])
			xc_pix, yc_pix = int(xc_rot/grid_res), int(yc_rot/ grid_res)
			rc_pix = (rc/ grid_res)
			inside_circle = ((X-xc_pix)**2 +(Y-yc_pix)**2 - rc_pix**2 < 0)
			occupied_pt = grid[X, Y] == 1
			if np.sum(np.multiply( inside_circle, occupied_pt)):
				return True
	return False

def track(grid,ref_path, pose, v, w, dt=0.1):
	commands = np.flipud(command_window(v, w, dt))
	pred_horizon,V_MAX,w_cte,w_speed = itemgetter('pred_horizon','V_MAX','w_cte','w_speed')(params)
	#initialize path cost
	best_cost, best_command = np.inf, None
	for i, (v, w) in enumerate(commands):
		local_path = simulate_unicycle(pose,v,w,pred_horizon) #Number of steps = prediction horizon
		if circle_collision_check(grid, local_path): #ignore colliding paths
			print(f'local path has a collision, best_command = {best_command}, cost = {best_cost}')
			continue

		#calculate cross-track error
		#can use a simplistic definition of 
		#how close is the last pose in local path from the ref path
		cte = euclidean(local_path[-1][:2],ref_path[-1][:2])

		#other cost functions are possible
		#can modify collision checker to give distance to closest obstacle
		cost = w_cte*cte + w_speed*(V_MAX - v)**2 # + w_obs/ distance

		#check if there is a better candidate
		if cost < best_cost:
			best_cost, best_command = cost, (v,w)

	if best_command:
		return best_command
	else:
		return [0, 0]
		
def plot_final_path(grid_dense,ref_path,logs,extent_limits):
	poses = np.array(logs)[:,:3]
	fig, ax = plt.subplots(figsize=(12,16))
	for s in ref_path[1:]:
		ax.plot(s[0], s[1],'r+')
	ax.plot(poses[:,0], poses[:,1],'.',c='g')
	ax.grid()
	ax.imshow(grid_dense, cmap=plt.cm.Dark2, extent=extent_limits) 
	
def plot_wvk(logs):
	plt.figure(figsize=(10,10))
	v = np.array(logs)[:,3]
	w = np.array(logs)[:,4]
	for i,x in enumerate(v):
		if x == 0:
			v[i] = 1e-3
	kappa = [x/y for x,y in zip(w,v)]
	plt.plot(w,label = 'w')
	plt.plot(v,label = 'v')
	plt.plot(kappa,label = 'k')
	plt.grid()
	plt.legend()