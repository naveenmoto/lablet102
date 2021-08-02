import time
from types import SimpleNamespace
import argparse


import cv2
import matplotlib.pyplot as plt
import numpy as np
import toml

from astar import Astar
from dwa import DWA

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--map", default="circuit", help="The name of the map. See config.toml for available maps.")

config = toml.load("config.toml")
config_params = config['params']
params = SimpleNamespace(**config_params)


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.
    img = np.flip(img, 0)
    return img


def downsample(img, grid_res=0.05):
    new_shape = (
        round(img.shape[1]*grid_res), round(img.shape[0]*grid_res))
    img_downsampled = cv2.resize(
        img, new_shape, cv2.INTER_NEAREST)
    return img_downsampled

args = parser.parse_args()
circuit_name = args.map

circuit, circuit_dynamic = config["maps"][circuit_name]

img = cv2.imread(f"./maps/{circuit}")
img = preprocess(img)

img_reality = cv2.imread(f"./maps/{circuit_dynamic}")
img_reality = preprocess(img_reality)

img_downsampled = downsample(img, grid_res=0.05)

# Making it binary
img_downsampled[img_downsampled > 0.3] = 1
img_downsampled[img_downsampled < 0.99] = 0

reality_downsampled = downsample(img_reality, grid_res=0.05)

# Making it binary
reality_downsampled[reality_downsampled > 0.3] = 1
reality_downsampled[reality_downsampled < 0.99] = 0

# Running A* algorithm
astar_ = Astar(1-img_downsampled)
start = config[circuit_name]["start"]
goal = config[circuit_name]["goal"]
path = astar_.shortest_path(start, goal)

x, y = path
interp_range = len(x)*2
x = np.interp(np.linspace(0, 1, interp_range), np.linspace(0, 1, len(x)), x)
y = np.interp(np.linspace(0, 1, interp_range), np.linspace(0, 1, len(y)), y)

# Calculate orientation
pre_x, pre_y = x[0], y[0]
t = [np.arctan2(y[1]-pre_y, x[1]-pre_x)]
for x_c, y_c in zip(x[1:], y[1:]):
    t.append(np.arctan2(y_c-pre_y, x_c-pre_x))
    pre_x, pre_y = x_c, y_c
t = np.array(t)


path = np.array([x, y, t], dtype=np.float).T

# Create DWA object
dwa_ = DWA(1-img_downsampled.T, path,
           path[0], goal_threshold=params.goal_threshold, reality=1-reality_downsampled.T)

plt.figure(figsize=(20, 20))

# Simulation loop
step = 0
for progress, distances, target_path in dwa_:

    tracked_x, tracked_y = progress[:, 0], progress[:, 1]
    plt.clf()
    plt.imshow(1-dwa_.grid_data.T, cmap="gray", origin="lower")
    plt.imshow(1-dwa_.reality.T, cmap="gray", origin="lower", alpha=0.5)
    plt.scatter(x, y, label="A* path")
    plt.plot(tracked_x, tracked_y, c="red", label="Tracked")
    idx_target = int(progress[-1, 5])

    x_target = x[idx_target: idx_target+params.pred_horizon]
    y_target = y[idx_target: idx_target+params.pred_horizon]

    plt.scatter(x_target, y_target, label="Prediction Horizon")
    x_c, y_c, theta_c = dwa_.pose
    for dist, angle in zip(distances, dwa_.lidar.beam_angles):
        t = angle + theta_c
        plt.plot(np.array([x_c, x_c+dist*np.cos(t)]),
                 np.array([y_c, y_c+dist*np.sin(t)]), c="green")
    if target_path is not None:
        plt.plot(target_path[:, 0], target_path[:, 1], label="Target path", linewidth=2, c="blue")

    plt.scatter(tracked_x[-1], tracked_y[-1], label="Robot", s=100, c="red", zorder=10)
    plt.legend()
    plt.pause(0.001)
    step += 1

    # Uncomment following lines if the simulation goes to an infinite loop
    # if step > 600:
    #     break

print("Simulation Finished!")
plt.legend()
plt.show()
