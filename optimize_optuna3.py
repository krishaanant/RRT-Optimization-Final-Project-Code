import optuna
import numpy as np
import traceback
import os
import matplotlib.pyplot as plt
    
from run_rrt import run_rrt
from envgen import generate_random_obstacles_pct

DEBUG = False
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
                coverage_percentage=OBSTACLE_COVERAGE_PERCENTAGE  # Fixed percentage coverage
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

    # If EVERY run failed → indicate that parameter set is bad
    if len(exec_times) == 0:
        return 9999.0, 9999.0, 1.0

    # Means for successful runs
    mean_exec_time = float(np.mean(exec_times))
    mean_path_cost = float(np.mean(path_costs))
    failure_rate = failures / float(n_repeats)

    return mean_exec_time, mean_path_cost, failure_rate


# -------------------------------------------------------
# Optuna multi-objective objective function
# -------------------------------------------------------
def objective(trial):
    params = {
        "step_size": trial.suggest_float("step_size", 0.5, 15.0),
        "goal_bias": trial.suggest_float("goal_bias", 0.0, 0.5),
    }

    # Extremely aggressive step size + high goal bias may blow up
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
# Parameter space visualization
# Shows step_size vs goal_bias
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
