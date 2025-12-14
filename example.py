from run_rrt import run_rrt
from envgen import *


# Example usage of run_rrt
if __name__ == "__main__":
    
    # Different random obstacle generators are available in envgen.py
    
    # obstacles = [(20, 20, 40, 40), (20, 60, 40, 80),
                #  (60, 20, 80, 40), (60, 60, 80, 80)]
   
    # Parameters
    step_size = 1
    goal_bias = 0.1

    # Constants
    X_dimensions = np.array([(0, 100), (0, 100)])
    start = (0, 0)
    goal = (100, 100)
    
    # Obstacle (Map) Generation
    # generate_random_obstacles_pct(dimension_lengths, start, end, coverage_percentage, tolerance=0.02)
    obstacle_coverage_percentage = 20  # percentage of workspace to be covered by obstacles
    obstacles = generate_random_obstacles_pct(X_dimensions, start, goal, obstacle_coverage_percentage)

    exec_time, path_cost, samples = run_rrt(
        obstacles,
        start,
        goal,
        step_size,
        goal_bias
    )

    print(f"Execution Time: {exec_time} seconds")
    print(f"Path Cost: {path_cost}")
    print(f"Samples Taken: {samples}")
