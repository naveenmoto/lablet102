import numpy as np
import matplotlib.pyplot as plt

def cubic_spiral(theta_i, theta_f, n=10):
    x = np.linspace(0, 1, num=n)
    #-2*x**3 + 3*x**2
    return (theta_f-theta_i)*(-2*x**3 + 3*x**2) + theta_i

def straight(dist, curr_pose, rad, n=10):
    x0,y0,t0 = curr_pose
    t = [t0]*n
    dn = dist/n
    x = x0 + np.cumsum(dn*np.cos(t))
    y = y0 + np.cumsum(dn*np.sin(t))
    return x,y,t

def turn(change, curr_pose, rad,dt=0.1, n=100):
    # adjust scaling constant for desired turn radius
	v = rad/6
	x0,y0,t0 = curr_pose
	t = cubic_spiral(t0, t0 + np.deg2rad(change), n)
	x= x0 + np.cumsum(v*dt*np.cos(t))
	y= y0 + np.cumsum(v*dt*np.sin(t))
	return x,y,t  

def generate_trajectory(route, turn_radius, init_pose = (0, 0,np.pi/2)):
	curr_pose = init_pose
	func = {'straight': straight, 'turn': turn}
	x, y, t = np.array([]), np.array([]),np.array([])
	for manoeuvre, command in route:
		px, py, pt = func[manoeuvre](command,curr_pose,turn_radius)
		curr_pose = (px[-1],py[-1],pt[-1])
		# update x, y, t using np.concatenate and px,py,pt
		x = np.concatenate([x,px])
		y = np.concatenate([y,py])
		t = np.concatenate([t,pt])
	return np.vstack((x,y,t))

def plot_smooth_route(grid, start,goal, astar_path,rx,ry,extent_limits):
	fig, ax = plt.subplots(figsize=(12,12))
	ax.imshow(np.flipud(grid), cmap=plt.cm.Dark2,origin='Lower',extent=extent_limits)
	ax.scatter(start[0],start[1], marker = "+", color = "yellow", s = 200)
	ax.scatter(goal[0],goal[1], marker = "+", color = "red", s = 200)
	for s in astar_path[1:]:
		ax.plot(s[0], s[1],'r+')	
	ax.plot(rx,ry,'blue')
	plt.show()