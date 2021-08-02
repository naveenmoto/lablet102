# THIS IS USING CUBIC SPIRAL FOR TURNING
# Using quintic spiral for turning results in almost the same turning trajectory
import numpy as np

v = 1
dt = 0.1
num_st_pts = int(v/dt)
num_pts = 50

def cubic_spiral(theta_i, theta_f, n=50):
    x = np.linspace(0, 1, num=n)
    #-2*x**3 + 3*x**2
    # return (theta_f-theta_i)*(6*x**5 - 15*x**4 + 10*x**3) + theta_i   # Use this for quintic plot if needed
    return (theta_f-theta_i)*(-2*x**3 + 3*x**2) + theta_i
    
def straight(dist, curr_pose, n=num_st_pts):
    # dist is the straight distance to be travelled
    # the straight-line may be along x or y axis
    # curr_theta will determine the orientation
    # print(n)
    x0, y0, t0 = curr_pose
    xf, yf = x0 + dist*np.cos(t0), y0 + dist*np.sin(t0)
    x = (xf - x0) * np.linspace(0, 1, n) + x0
    y = (yf - y0) * np.linspace(0, 1, n) + y0
    return x, y, t0*np.ones_like(x)

def turn(change, curr_pose, n=num_pts):
     # adjust scaling constant for desired turn radius
     # change is the angle by which we want to turn, like 90 or -90
    #  print(n)
     x0, y0, t0 = curr_pose
     theta = cubic_spiral(t0, t0 + np.deg2rad(change), n)
     # cumsum() is basically integration
     x= x0 + np.cumsum(v*np.cos(theta)*dt)
     y= y0 + np.cumsum(v*np.sin(theta)*dt)
     return x, y, theta

def generate_trajectory(route, init_pose = (0, 0,np.pi/2)):
    curr_pose = init_pose
    func = {'straight': straight, 'turn': turn}
    x, y, t = np.array([]), np.array([]),np.array([])
    for manoeuvre, command in route:
        px, py, pt = func[manoeuvre](command, curr_pose)
        curr_pose = px[-1],py[-1],pt[-1]  # Pull last element of px, py, pt to get latest orientation
        # print(np.round(px[-1],2),np.round(py[-1],2),np.round(pt[-1],2))   # This is the final position after every route iteration
        # Add latest pose to array
        x = np.concatenate([x, px])
        y = np.concatenate([y, py])
        t = np.concatenate([t, pt]) 
    # print(x[-1], y[-1], t[-1])   # This is the final position
    return np.vstack([x, y, t])