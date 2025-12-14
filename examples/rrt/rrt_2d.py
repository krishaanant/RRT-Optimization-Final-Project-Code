# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np

from rrt_algorithms.rrt.rrt import RRT
from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.plotting import Plot

# Configuration Space
X_dimensions = np.array([(0, 100), (0, 100)])  # dimensions of Search Space

# Inputs
Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80),
                     (60, 20, 80, 40), (60, 60, 80, 80)])
x_init = (0, 0)  # starting location
x_goal = (100, 100)  # goal location

# Parameters
step_size = 5  # length of tree edges
goal_bias = 0.05  # probability of sampling the goal directly

# Other constants
r = 0.1  # length of smallest edge to check for intersection with obstacles
max_samples = 5000  # max number of samples to take before timing out
prc = 0.1  # probability of checking for a connection to goal

# create search space
X = SearchSpace(X_dimensions, Obstacles, x_goal, goal_bias=goal_bias)

# create rrt_search
rrt = RRT(X, step_size, x_init, x_goal, max_samples, r, prc)
path = rrt.rrt_search()

# # plot
# plot = Plot("rrt_2d")
# plot.plot_tree(X, rrt.trees)
# if path is not None:
#     plot.plot_path(X, path)
# plot.plot_obstacles(X, Obstacles)
# plot.plot_start(X, x_init)
# plot.plot_goal(X, x_goal)
# plot.draw(auto_open=True)
