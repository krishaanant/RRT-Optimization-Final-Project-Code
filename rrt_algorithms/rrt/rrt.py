from rrt_algorithms.rrt.rrt_base import RRTBase
import time
import math


class RRT(RRTBase):
    def __init__(self, X, q, x_init, x_goal, max_samples, r, prc=0.01):
        """
        Template RRT planner
        :param X: Search Space
        :param q: list of lengths of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        """
        super().__init__(X, q, x_init, x_goal, max_samples, r, prc)
        # metrics set during/after search
        self._execution_time = None  # seconds
        self.solution_path = None

    def rrt_search(self):
        """
        Create and return a Rapidly-exploring Random Tree, keeps expanding until can connect to goal
        https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree
        :return: list representation of path, dict representing edges of tree in form E[child] = parent
        """
        # start timing
        start = time.perf_counter()

        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)

        while True:
            x_new, x_nearest = self.new_and_near(0, self.q)

            if x_new is None:
                continue

            # connect shortest valid edge
            self.connect_to_point(0, x_nearest, x_new)

            solution = self.check_solution()
            if solution[0]:
                # record solution and execution time
                self.solution_path = solution[1]
                self._execution_time = time.perf_counter() - start
                return solution[1]

    def get_execution_time(self):
        """Return execution time (in seconds) of the last rrt_search run, or None if not run."""
        return self._execution_time

    def get_tree_size(self):
        """Return total number of nodes across all trees maintained by the planner."""
        try:
            return sum(t.V_count for t in self.trees)
        except Exception:
            # fallback: 0 if trees not initialized
            return 0
        
    def get_samples_taken(self):
        """Return total number of samples taken during the last rrt_search run."""
        try:
            return self.samples_taken
        except Exception:
            # fallback: 0 if trees not initialized
            return 0

    def get_path_cost(self):
        """Return path cost (sum of Euclidean edge lengths) for the found solution path, or None if no solution."""
        path = self.solution_path
        if not path:
            return None
        # sum euclidean distances between consecutive waypoints
        total = 0.0
        for a, b in zip(path[:-1], path[1:]):
            try:
                total += math.dist(a, b)
            except AttributeError:
                # math.dist may not be available; compute manually
                total += math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
        return total

