import numpy as np


def simulate_unicycle(pose, v, w, N=1, dt=0.1):
    x, y, t = pose
    poses = []
    for _ in range(N):
        x += v*np.cos(t)*dt
        y += v*np.sin(t)*dt
        t += w*dt
        poses.append([x, y, t])
    return np.array(poses)
