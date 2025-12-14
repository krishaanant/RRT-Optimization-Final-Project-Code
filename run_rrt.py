"""Utility runner for quickly running RRT and returning metrics.

This module provides a single function `run_rrt` which hides construction
details for the RRT planner and returns three costs:
 - execution time (seconds)
 - path cost (sum of Euclidean edge lengths)
 - tree size (number of nodes)

The function accepts an obstacle array (or None), start and goal tuples,
the step size `q` and a `goal_bias` in [0,1]. Other planner parameters have
reasonable defaults but may be passed through.
"""
from typing import Optional, Sequence, Tuple
import numpy as np

from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.rrt.rrt import RRT


def run_rrt(obstacles: Optional[Sequence],
            start: Tuple[float, ...],
            goal: Tuple[float, ...],
            step_size: float,
            goal_bias: float,
            max_samples: int = 10000,
            r: float = 1,
            prc: float = 0.01) -> Tuple[Optional[float], Optional[float], int]:
    """Run RRT with minimal caller inputs and return (exec_time, path_cost, tree_size).

    Parameters
    - obstacles: sequence of obstacles in the same format used by the examples
                 (e.g., list/array of (x1,y1,x2,y2) for 2D). Pass None for empty.
    - start, goal: coordinate tuples
    - step_size: step length used by the planner
    - goal_bias: probability in [0,1] of sampling the goal directly
    - max_samples, r, prc: optional planner parameters
    - padding: how much to pad the computed bounding box

    Returns (execution_time_seconds OR None, path_cost OR None, tree_size_int)
    """
    X_dimensions = np.array([(0, 100), (0, 100)])  # dimensions of Search Space

    # create SearchSpace. If obstacles is empty/None pass None
    X = SearchSpace(X_dimensions, goal, goal_bias=goal_bias, O=obstacles)

    # create planner
    planner = RRT(X, step_size, start, goal, max_samples, r, prc)

    # run planner
    path = planner.rrt_search()

    exec_time = planner.get_execution_time() if hasattr(planner, 'get_execution_time') else None
    path_cost = planner.get_path_cost() if hasattr(planner, 'get_path_cost') else None
    samples = planner.get_samples_taken() if hasattr(planner, 'get_samples_taken') else sum(t.V_count for t in planner.trees)

    return exec_time, path_cost, samples