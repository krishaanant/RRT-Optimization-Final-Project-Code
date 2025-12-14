#!/usr/bin/env python3
"""
optimize_optuna3.py

Multi-objective optimization of RRT parameters using Optuna.
Objectives: [execution_time, path_cost, failure_rate]

This version:
- Keeps your previous behavior (3 repeats per evaluation).
- Handles RRT crashes robustly and returns large penalty values.
- Allows silencing stack traces with DEBUG flag.
- Adds a small heuristic to immediately penalize obviously bad parameter combos.
"""
'''
import optuna
import numpy as np
import traceback
import os
import matplotlib.pyplot as plt
    
from run_rrt import run_rrt
from envgen import generate_random_obstacles_pct


# -------------------------------------------------------
# Global debug flag (set to True if you want detailed crash logs)
# -------------------------------------------------------
DEBUG = False


# -------------------------------------------------------
# Objective wrapper with repeated RRT evaluations
# -------------------------------------------------------
def evaluate_params(params, n_repeats=3, max_failures=None):
    """
    Run RRT multiple times with the same parameter set on different random maps.

    Returns:
        mean_exec_time, mean_path_cost, failure_rate
    """
    if max_failures is None:
        max_failures = n_repeats  # allow all to fail before bailing

    exec_times = []
    path_costs = []
    failures = 0

    for rep in range(n_repeats):
        try:
            # Generate a fresh random environment per repeat
            X_dimensions = np.array([(0, 100), (0, 100)])
            start = (0, 0)
            goal = (100, 100)

            obstacles = generate_random_obstacles_pct(
                X_dimensions,
                start,
                goal,
                coverage_percentage=20
            )

            exec_time, path_cost, samples = run_rrt(
                obstacles,
                start,
                goal,
                step_size=params["step_size"],
                goal_bias=params["goal_bias"],
                max_samples=params["max_samples"],
                r=params["radius"],
                prc=params["prc"]
            )

            # If RRT returned invalid values, treat this as a failure
            if exec_time is None or path_cost is None:
                failures += 1
                if failures >= max_failures:
                    break
                continue

            exec_times.append(exec_time)
            path_costs.append(path_cost)

        except Exception as e:
            failures += 1
            if DEBUG:
                print(f"[Warning] RRT crashed on repeat {rep+1}/{n_repeats} with params={params}")
                print("Error:", e)
                print(traceback.format_exc())
            if failures >= max_failures:
                break

    # If EVERY run failed → this parameter set is terrible
    if len(exec_times) == 0:
        return 9999.0, 9999.0, 1.0

    # Compute means for successful runs
    mean_exec_time = float(np.mean(exec_times))
    mean_path_cost = float(np.mean(path_costs))
    failure_rate = failures / float(n_repeats)

    return mean_exec_time, mean_path_cost, failure_rate


# -------------------------------------------------------
# Optuna multi-objective objective function
# -------------------------------------------------------
def objective(trial):
    # Sample RRT hyperparameters
    params = {
        "step_size": trial.suggest_float("step_size", 0.5, 15.0),
        "goal_bias": trial.suggest_float("goal_bias", 0.0, 0.5),
        "max_samples": trial.suggest_int("max_samples", 300, 5000),
        "radius": trial.suggest_float("radius", 1.0, 10.0),
        "prc": trial.suggest_float("prc", 0.001, 0.1),
    }

    # ------------------------------------------------------------------
    # Optional heuristics: instantly penalize obviously problematic zones
    # (you can comment these out if you don't like them)
    # ------------------------------------------------------------------
    # Example: very large radius + tiny collision-check step can be unstable
    if params["radius"] > 9.0 and params["prc"] < 0.01:
        return 9999.0, 9999.0, 1.0

    # Example: extremely aggressive step size + high goal bias may blow up
    if params["step_size"] > 13.0 and params["goal_bias"] > 0.3:
        return 9999.0, 9999.0, 1.0
    # ------------------------------------------------------------------

    exec_time, path_cost, failure_rate = evaluate_params(params, n_repeats=3)
    return exec_time, path_cost, failure_rate


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    print("=============================================")
    print("     MULTI-OBJECTIVE OPTUNA OPTIMIZATION     ")
    print("   Objectives = [execution_time, path_cost, failure_rate]")
    print("=============================================\n")

    # Optional: set a reproducible sampler
    sampler = optuna.samplers.NSGAIISampler(seed=42)

    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
        sampler=sampler,
    )

    # Run optimization
    study.optimize(objective, n_trials=60, n_jobs=1)

    print("\n====================")
    print(" Optimization Done! ")
    print("====================")

    # Pareto-optimal trials (multi-objective front)
    print("\nPareto-optimal solutions:")
    for sol in study.best_trials:
        print("\nTrial:", sol.number)
        print("Values  =", sol.values)   # [exec_time, path_cost, failure_rate]
        print("Params  =", sol.params)

    # -------------------------------------------------------
    # Save Pareto scatter plot (Execution Time vs Path Cost)
    # Colored by Failure Rate
    # -------------------------------------------------------
    exec_times = [t.values[0] for t in study.best_trials]
    path_costs = [t.values[1] for t in study.best_trials]
    failure_rates = [t.values[2] for t in study.best_trials]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(exec_times, path_costs, c=failure_rates, cmap="viridis", s=60)
    cbar = plt.colorbar(sc)
    cbar.set_label("Failure Rate")

    plt.xlabel("Execution Time")
    plt.ylabel("Path Cost")
    plt.title("Pareto Frontier (Colored by Failure Rate)")
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "pareto_frontier.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    print(f"\nSaved plot to {out_path}")
'''

#!/usr/bin/env python3
"""
optimize_optuna3.py

Multi-objective optimization of RRT parameters using Optuna.
Objectives: [execution_time, path_cost, failure_rate]

This version:
- Uses 30% obstacle coverage for all evaluations
- Runs N=20 simulations per trial and averages results
- Uses correct parameters for run_rrt function
- Handles RRT crashes robustly and returns large penalty values
- Allows silencing stack traces with DEBUG flag
"""
'''
import optuna
import numpy as np
import traceback
import os
import matplotlib.pyplot as plt
    
from run_rrt import run_rrt
from envgen import generate_random_obstacles_pct


# -------------------------------------------------------
# Global debug flag (set to True if you want detailed crash logs)
# -------------------------------------------------------
DEBUG = False

# -------------------------------------------------------
# Configuration constants
# -------------------------------------------------------
OBSTACLE_COVERAGE_PERCENTAGE = 30  # Fixed at 30% obstacle coverage
SIMULATIONS_PER_TRIAL = 20  # N=20 simulations per trial


# -------------------------------------------------------
# Objective wrapper with repeated RRT evaluations
# -------------------------------------------------------
def evaluate_params(params, n_repeats=SIMULATIONS_PER_TRIAL, max_failures=None):
    """
    Run RRT multiple times with the same parameter set on different random maps.

    Returns:
        mean_exec_time, mean_path_cost, failure_rate
    """
    if max_failures is None:
        max_failures = n_repeats  # allow all to fail before bailing

    exec_times = []
    path_costs = []
    failures = 0

    for rep in range(n_repeats):
        try:
            # Generate a fresh random environment per repeat
            X_dimensions = np.array([(0, 100), (0, 100)])
            start = (0, 0)
            goal = (100, 100)

            obstacles = generate_random_obstacles_pct(
                X_dimensions,
                start,
                goal,
                coverage_percentage=OBSTACLE_COVERAGE_PERCENTAGE  # Fixed 30% coverage
            )

            # Call run_rrt with correct parameters based on your example
            exec_time, path_cost, samples = run_rrt(
                obstacles,
                start,
                goal,
                step_size=params["step_size"],
                goal_bias=params["goal_bias"]
                # Note: max_samples, radius, and prc are removed since they're not in your function
            )

            # If RRT returned invalid values, treat this as a failure
            if exec_time is None or path_cost is None:
                failures += 1
                if failures >= max_failures:
                    break
                continue

            exec_times.append(exec_time)
            path_costs.append(path_cost)

        except Exception as e:
            failures += 1
            if DEBUG:
                print(f"[Warning] RRT crashed on repeat {rep+1}/{n_repeats} with params={params}")
                print("Error:", e)
                print(traceback.format_exc())
            if failures >= max_failures:
                break

    # If EVERY run failed → this parameter set is terrible
    if len(exec_times) == 0:
        return 9999.0, 9999.0, 1.0

    # Compute means for successful runs
    mean_exec_time = float(np.mean(exec_times))
    mean_path_cost = float(np.mean(path_costs))
    failure_rate = failures / float(n_repeats)

    return mean_exec_time, mean_path_cost, failure_rate


# -------------------------------------------------------
# Optuna multi-objective objective function
# -------------------------------------------------------
def objective(trial):
    # Sample only the parameters that your run_rrt function actually uses
    params = {
        "step_size": trial.suggest_float("step_size", 0.5, 15.0),
        "goal_bias": trial.suggest_float("goal_bias", 0.0, 0.5),
        # Removed: max_samples, radius, prc since they're not in your run_rrt function
    }

    # ------------------------------------------------------------------
    # Optional heuristics: you can adjust these based on your actual parameters
    # ------------------------------------------------------------------
    # Example: extremely aggressive step size + high goal bias may blow up
    if params["step_size"] > 13.0 and params["goal_bias"] > 0.3:
        return 9999.0, 9999.0, 1.0
    # ------------------------------------------------------------------

    exec_time, path_cost, failure_rate = evaluate_params(params, n_repeats=SIMULATIONS_PER_TRIAL)
    return exec_time, path_cost, failure_rate


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    print("=============================================")
    print("     MULTI-OBJECTIVE OPTUNA OPTIMIZATION     ")
    print("   Objectives = [execution_time, path_cost, failure_rate]")
    print(f"   Obstacle Coverage = {OBSTACLE_COVERAGE_PERCENTAGE}%")
    print(f"   Simulations per trial = {SIMULATIONS_PER_TRIAL}")
    print("   Parameters: step_size, goal_bias")
    print("=============================================\n")

    # Optional: set a reproducible sampler
    sampler = optuna.samplers.NSGAIISampler(seed=42)

    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
        sampler=sampler,
    )

    # Run optimization
    study.optimize(objective, n_trials=60, n_jobs=1)

    print("\n====================")
    print(" Optimization Done! ")
    print("====================")

    # Pareto-optimal trials (multi-objective front)
    print(f"\nPareto-optimal solutions (coverage={OBSTACLE_COVERAGE_PERCENTAGE}%, N={SIMULATIONS_PER_TRIAL}):")
    for sol in study.best_trials:
        print("\nTrial:", sol.number)
        print("Values  =", sol.values)   # [exec_time, path_cost, failure_rate]
        print("Params  =", sol.params)

    # -------------------------------------------------------
    # Save Pareto scatter plot (Execution Time vs Path Cost)
    # Colored by Failure Rate
    # -------------------------------------------------------
    exec_times = [t.values[0] for t in study.best_trials]
    path_costs = [t.values[1] for t in study.best_trials]
    failure_rates = [t.values[2] for t in study.best_trials]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(exec_times, path_costs, c=failure_rates, cmap="viridis", s=60)
    cbar = plt.colorbar(sc)
    cbar.set_label("Failure Rate")

    plt.xlabel("Execution Time")
    plt.ylabel("Path Cost")
    plt.title(f"Pareto Frontier ({OBSTACLE_COVERAGE_PERCENTAGE}% Obstacles, N={SIMULATIONS_PER_TRIAL})")
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "pareto_frontier.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    print(f"\nSaved plot to {out_path}")

    # -------------------------------------------------------
# ADDITIONAL PLOT: Parameter space visualization
# Shows what you actually optimized: step_size vs goal_bias
# -------------------------------------------------------
print("\nGenerating parameter space plot...")

# Collect all trial data
all_trials = study.trials

# Extract parameter values and execution times
step_sizes = [t.params.get('step_size', np.nan) for t in all_trials]
goal_biases = [t.params.get('goal_bias', np.nan) for t in all_trials]
exec_times = [t.values[0] if t.values else np.nan for t in all_trials]

# Create the plot
plt.figure(figsize=(10, 8))

# Plot all trials colored by execution time
sc = plt.scatter(step_sizes, goal_biases, c=exec_times, 
                 cmap='viridis', s=80, alpha=0.8, edgecolors='black', linewidth=0.5)

# Highlight Pareto-optimal solutions
pareto_step_sizes = [t.params.get('step_size', np.nan) for t in study.best_trials]
pareto_goal_biases = [t.params.get('goal_bias', np.nan) for t in study.best_trials]
plt.scatter(pareto_step_sizes, pareto_goal_biases, 
            color='red', s=120, marker='*', edgecolors='black', 
            linewidth=1.5, label='Pareto-optimal')

plt.colorbar(sc, label='Execution Time')
plt.xlabel('Step Size (optimized parameter)')
plt.ylabel('Goal Bias (optimized parameter)')
plt.title(f'Optimization Results: Step Size vs Goal Bias\n'
          f'({OBSTACLE_COVERAGE_PERCENTAGE}% obstacles, N={SIMULATIONS_PER_TRIAL} simulations)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
param_plot_path = os.path.join("results", "optimized_parameters.png")
plt.savefig(param_plot_path, dpi=150, bbox_inches="tight")
print(f"Saved parameter space plot to {param_plot_path}")

#!/usr/bin/env python3
'''
"""
optimize_optuna3.py

Multi-objective optimization of RRT parameters using Optuna.
Objectives: [execution_time, path_cost, failure_rate]

This version:
- Uses 30% obstacle coverage for all evaluations
- Runs N=20 simulations per trial and averages results
- Optimizes ONLY step_size parameter
"""

import sys
import os
import optuna
import numpy as np
import traceback
import matplotlib.pyplot as plt

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from run_rrt import run_rrt
    from envgen import generate_random_obstacles_pct
    print("Successfully imported run_rrt and envgen")
except ImportError as e:
    print(f"Import error: {e}")
    print("Current directory:", os.getcwd())
    print("Files in directory:", [f for f in os.listdir('.') if f.endswith('.py')])
    sys.exit(1)


# -------------------------------------------------------
# Global debug flag (set to True if you want detailed crash logs)
# -------------------------------------------------------
DEBUG = False

# -------------------------------------------------------
# Configuration constants
# -------------------------------------------------------
OBSTACLE_COVERAGE_PERCENTAGE = 30  # Fixed at 30% obstacle coverage
SIMULATIONS_PER_TRIAL = 20  # N=20 simulations per trial
GOAL_BIAS = 0.1  # Fixed goal_bias value


# -------------------------------------------------------
# Objective wrapper with repeated RRT evaluations
# -------------------------------------------------------
def evaluate_params(params, n_repeats=SIMULATIONS_PER_TRIAL, max_failures=None):
    """
    Run RRT multiple times with the same parameter set on different random maps.

    Returns:
        mean_exec_time, mean_path_cost, failure_rate
    """
    if max_failures is None:
        max_failures = n_repeats  # allow all to fail before bailing

    exec_times = []
    path_costs = []
    failures = 0

    for rep in range(n_repeats):
        try:
            # Generate a fresh random environment per repeat
            X_dimensions = np.array([(0, 100), (0, 100)])
            start = (0, 0)
            goal = (100, 100)

            obstacles = generate_random_obstacles_pct(
                X_dimensions,
                start,
                goal,
                coverage_percentage=OBSTACLE_COVERAGE_PERCENTAGE  # Fixed 30% coverage
            )

            # Call run_rrt with step_size from params and fixed goal_bias
            exec_time, path_cost, samples = run_rrt(
                obstacles,
                start,
                goal,
                step_size=params["step_size"],
                goal_bias=GOAL_BIAS  # Fixed value
            )

            # If RRT returned invalid values, treat this as a failure
            if exec_time is None or path_cost is None:
                failures += 1
                if failures >= max_failures:
                    break
                continue

            exec_times.append(exec_time)
            path_costs.append(path_cost)

        except Exception as e:
            failures += 1
            if DEBUG:
                print(f"[Warning] RRT crashed on repeat {rep+1}/{n_repeats} with params={params}")
                print("Error:", e)
                print(traceback.format_exc())
            if failures >= max_failures:
                break

    # If EVERY run failed → this parameter set is terrible
    if len(exec_times) == 0:
        return 9999.0, 9999.0, 1.0

    # Compute means for successful runs
    mean_exec_time = float(np.mean(exec_times))
    mean_path_cost = float(np.mean(path_costs))
    failure_rate = failures / float(n_repeats)

    return mean_exec_time, mean_path_cost, failure_rate


# -------------------------------------------------------
# Optuna multi-objective objective function
# -------------------------------------------------------
def objective(trial):
    # Sample ONLY step_size parameter
    params = {
        "step_size": trial.suggest_float("step_size", 0.5, 15.0),
        # goal_bias is now fixed, not optimized
    }

    exec_time, path_cost, failure_rate = evaluate_params(params, n_repeats=SIMULATIONS_PER_TRIAL)
    return exec_time, path_cost, failure_rate


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    print("=============================================")
    print("     MULTI-OBJECTIVE OPTUNA OPTIMIZATION     ")
    print("   Objectives = [execution_time, path_cost, failure_rate]")
    print(f"   Obstacle Coverage = {OBSTACLE_COVERAGE_PERCENTAGE}%")
    print(f"   Simulations per trial = {SIMULATIONS_PER_TRIAL}")
    print(f"   Fixed goal_bias = {GOAL_BIAS}")
    print("   Optimizing: step_size ONLY")
    print("=============================================\n")

    # Optional: set a reproducible sampler
    sampler = optuna.samplers.NSGAIISampler(seed=42)

    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
        sampler=sampler,
    )

    # Run optimization
    study.optimize(objective, n_trials=60, n_jobs=1)

    print("\n====================")
    print(" Optimization Done! ")
    print("====================")

    # Pareto-optimal trials (multi-objective front)
    print(f"\nPareto-optimal solutions (coverage={OBSTACLE_COVERAGE_PERCENTAGE}%, N={SIMULATIONS_PER_TRIAL}):")
    for sol in study.best_trials:
        print("\nTrial:", sol.number)
        print("Values  =", sol.values)   # [exec_time, path_cost, failure_rate]
        print("Params  =", sol.params)

    # Save plot
    exec_times = [t.values[0] for t in study.best_trials]
    path_costs = [t.values[1] for t in study.best_trials]
    failure_rates = [t.values[2] for t in study.best_trials]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(exec_times, path_costs, c=failure_rates, cmap="viridis", s=60)
    cbar = plt.colorbar(sc)
    cbar.set_label("Failure Rate")

    plt.xlabel("Execution Time")
    plt.ylabel("Path Cost")
    plt.title(f"Pareto Frontier ({OBSTACLE_COVERAGE_PERCENTAGE}% Obstacles, N={SIMULATIONS_PER_TRIAL}, goal_bias={GOAL_BIAS})")
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "pareto_frontier_step.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    print(f"\nSaved plot to {out_path}")

# -------------------------------------------------------
# ADDITIONAL SINGLE PLOT: Step_size optimization results
# -------------------------------------------------------
print("\nGenerating step_size optimization plot...")

# Collect all trial data
all_trials = study.trials

# Extract step_size values and mark Pareto-optimal ones
step_sizes = []
is_pareto = []

for trial in all_trials:
    if trial.values:
        step_sizes.append(trial.params.get('step_size', np.nan))
        # Check if this trial is Pareto-optimal
        is_pareto.append(trial in study.best_trials)

# Convert to arrays
step_sizes = np.array(step_sizes)
is_pareto = np.array(is_pareto)

# Create the SINGLE plot
plt.figure(figsize=(10, 8))

# Plot all step_size values
# Regular trials in blue
plt.scatter(step_sizes[~is_pareto], np.zeros_like(step_sizes[~is_pareto]), 
            color='blue', s=80, alpha=0.6, label='All trials')

# Pareto-optimal trials in red stars
plt.scatter(step_sizes[is_pareto], np.zeros_like(step_sizes[is_pareto]), 
            color='red', s=200, marker='*', edgecolors='black', 
            linewidth=1.5, label='Pareto-optimal')

# Add a horizontal line for reference
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)

# Add text labels for the range
plt.text(np.min(step_sizes), 0.02, f'min={np.min(step_sizes):.2f}', 
         fontsize=10, ha='left', color='gray')
plt.text(np.max(step_sizes), 0.02, f'max={np.max(step_sizes):.2f}', 
         fontsize=10, ha='right', color='gray')

plt.xlabel('Step Size (parameter being optimized)', fontsize=12)
plt.yticks([])  # Hide y-axis ticks since we're just showing positions
plt.title(f'Step Size Optimization Results\n'
          f'Goal Bias = {GOAL_BIAS} (fixed), {OBSTACLE_COVERAGE_PERCENTAGE}% obstacles\n'
          f'{SIMULATIONS_PER_TRIAL} simulations per trial, {len(step_sizes)} trials total',
          fontsize=14)
plt.grid(True, alpha=0.3, axis='x')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.tight_layout()

single_plot_path = os.path.join("results", "step_size_optimization_results.png")
plt.savefig(single_plot_path, dpi=150, bbox_inches="tight")
print(f"Saved SINGLE step_size optimization plot to {single_plot_path}")
#!/usr/bin/env python3
"""
optimize_optuna3.py

Multi-objective optimization of RRT parameters using Optuna.
Objectives: [execution_time, path_cost, failure_rate]

This version only optimizes goal_bias (step_size is fixed)
"""
'''
import sys
import os
import optuna
import numpy as np
import traceback
import matplotlib.pyplot as plt

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from run_rrt import run_rrt
    from envgen import generate_random_obstacles_pct
    print("Successfully imported run_rrt and envgen")
except ImportError as e:
    print(f"Import error: {e}")
    print("Current directory:", os.getcwd())
    print("Files in directory:", [f for f in os.listdir('.') if f.endswith('.py')])
    sys.exit(1)


# -------------------------------------------------------
# Global debug flag (set to True if you want detailed crash logs)
# -------------------------------------------------------
DEBUG = False

# -------------------------------------------------------
# Configuration constants
# -------------------------------------------------------
OBSTACLE_COVERAGE_PERCENTAGE = 30  # Fixed at 30% obstacle coverage
SIMULATIONS_PER_TRIAL = 20  # N=20 simulations per trial
FIXED_STEP_SIZE = 5.0  # Fixed step_size value (you can change this)


# -------------------------------------------------------
# Objective wrapper with repeated RRT evaluations
# -------------------------------------------------------
def evaluate_params(params, n_repeats=SIMULATIONS_PER_TRIAL, max_failures=None):
    """
    Run RRT multiple times with the same parameter set on different random maps.

    Returns:
        mean_exec_time, mean_path_cost, failure_rate
    """
    if max_failures is None:
        max_failures = n_repeats  # allow all to fail before bailing

    exec_times = []
    path_costs = []
    failures = 0

    for rep in range(n_repeats):
        try:
            # Generate a fresh random environment per repeat
            X_dimensions = np.array([(0, 100), (0, 100)])
            start = (0, 0)
            goal = (100, 100)

            obstacles = generate_random_obstacles_pct(
                X_dimensions,
                start,
                goal,
                coverage_percentage=OBSTACLE_COVERAGE_PERCENTAGE  # Fixed 30% coverage
            )

            # Call run_rrt with fixed step_size and optimized goal_bias
            exec_time, path_cost, samples = run_rrt(
                obstacles,
                start,
                goal,
                step_size=FIXED_STEP_SIZE,  # Fixed value
                goal_bias=params["goal_bias"]  # Only this is being optimized
            )

            # If RRT returned invalid values, treat this as a failure
            if exec_time is None or path_cost is None:
                failures += 1
                if failures >= max_failures:
                    break
                continue

            exec_times.append(exec_time)
            path_costs.append(path_cost)

        except Exception as e:
            failures += 1
            if DEBUG:
                print(f"[Warning] RRT crashed on repeat {rep+1}/{n_repeats} with params={params}")
                print("Error:", e)
                print(traceback.format_exc())
            if failures >= max_failures:
                break

    # If EVERY run failed → this parameter set is terrible
    if len(exec_times) == 0:
        return 9999.0, 9999.0, 1.0

    # Compute means for successful runs
    mean_exec_time = float(np.mean(exec_times))
    mean_path_cost = float(np.mean(path_costs))
    failure_rate = failures / float(n_repeats)

    return mean_exec_time, mean_path_cost, failure_rate


# -------------------------------------------------------
# Optuna multi-objective objective function
# -------------------------------------------------------
def objective(trial):
    # Only optimize goal_bias (step_size is fixed)
    params = {
        "goal_bias": trial.suggest_float("goal_bias", 0.0, 0.5),
        # step_size is fixed to FIXED_STEP_SIZE
    }

    exec_time, path_cost, failure_rate = evaluate_params(params, n_repeats=SIMULATIONS_PER_TRIAL)
    return exec_time, path_cost, failure_rate


# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":
    print("=============================================")
    print("     MULTI-OBJECTIVE OPTUNA OPTIMIZATION     ")
    print("   Objectives = [execution_time, path_cost, failure_rate]")
    print(f"   Obstacle Coverage = {OBSTACLE_COVERAGE_PERCENTAGE}%")
    print(f"   Simulations per trial = {SIMULATIONS_PER_TRIAL}")
    print(f"   Fixed step_size = {FIXED_STEP_SIZE}")
    print("   Optimizing: goal_bias")
    print("=============================================\n")

    # Optional: set a reproducible sampler
    sampler = optuna.samplers.NSGAIISampler(seed=42)

    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
        sampler=sampler,
    )

    # Run optimization
    study.optimize(objective, n_trials=60, n_jobs=1)

    print("\n====================")
    print(" Optimization Done! ")
    print("====================")

    # Pareto-optimal trials (multi-objective front)
    print(f"\nPareto-optimal solutions:")
    print(f"Coverage={OBSTACLE_COVERAGE_PERCENTAGE}%, N={SIMULATIONS_PER_TRIAL}, step_size={FIXED_STEP_SIZE}")
    for sol in study.best_trials:
        print("\nTrial:", sol.number)
        print("Values  =", sol.values)   # [exec_time, path_cost, failure_rate]
        print("Params  =", sol.params)

    # Save plot
    exec_times = [t.values[0] for t in study.best_trials]
    path_costs = [t.values[1] for t in study.best_trials]
    failure_rates = [t.values[2] for t in study.best_trials]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(exec_times, path_costs, c=failure_rates, cmap="viridis", s=60)
    cbar = plt.colorbar(sc)
    cbar.set_label("Failure Rate")

    plt.xlabel("Execution Time")
    plt.ylabel("Path Cost")
    plt.title(f"Pareto Frontier (step_size={FIXED_STEP_SIZE}, {OBSTACLE_COVERAGE_PERCENTAGE}% Obstacles)")
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "pareto_frontier_bias.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    print(f"\nSaved plot to {out_path}")

'''  
'''
OPTIMIZED_PARAMS = {
    "step_size": 1.78,  # Replace with your actual best step_size 
    "goal_bias": 0.098   # Replace with your actual best goal_bias
}

# Baseline parameters for comparison
BASELINE_PARAMS = {
    "step_size": 5.0,   # Typical default
    "goal_bias": 0.1   # Typical default
}

'''