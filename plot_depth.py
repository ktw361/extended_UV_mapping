import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
from mpl_toolkits.mplot3d import Axes3D

from visualize import Scene

import odlib

import os
import os.path as osp

base = '/home/damon/DATASETS/fat/single/002_master_chef_can_16k/kitchen_0/'

with open(osp.join(base, '_camera_settings.json')) as fp:
    camera_settings = json.load(fp)
with open(osp.join(base, '_object_settings.json')) as fp:
    object_settings = json.load(fp)

scene = Scene(base, 95)

rate = 0.1

depth = np.mean(scene.left.depth, 2)
h, w = depth.shape
depth = cv2.resize(depth, (int(rate*w), int(rate*h)))

h, w = depth.shape
x, y = np.meshgrid(np.arange(w), np.arange(h))
d = depth

fig = plt.figure()
ax = Axes3D(fig)
x = x.reshape(-1)
y = y.reshape(-1)
d = d.reshape(-1)
ax.scatter(x, y, d, c='b', marker='^', s=1)
plt.show()
