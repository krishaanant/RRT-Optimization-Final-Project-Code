## Team Contributions

This project was completed collaboratively. The specific responsibilities and contributions of each team member are outlined below to clearly document the partition of work:

- **Shubham**
  - Worked on the **background section**, providing overall context and motivation for the project.
  - Contributed to **related theory and practice**, covering foundational concepts and prior work.

- **Kellan**
  - Contributed to **related theory and practice**.
  - Led the explanation and presentation of the **Tree-structured Parzen Estimator (TPE)** algorithm.

- **Mijung**
  - Proposed and developed the **new idea**, including the **comparison of optimization algorithms using Optuna**.
  - Prepared and presented relevant slides associated with this contribution.

- **Krisha**
  - Implemented the core system and experiments.
  - Documented **implementation details**, including the COCO-based setup and methodology.

- **Ashutosh**
  - Conducted the experiments and analyzed outcomes.
  - Prepared the **experimental results** section and contributed to the conclusions.

- **All Team Members**
  - Participated in discussions, iterative refinement of the project, review of results, and final report preparation.

# RRT-Optimization-Final-Project-Code

Explanation of files and folders:
\\\\
rrt_algorithms/ — RRT implementation and supporting utilities 
GitHub

examples/rrt/ — examples and/or demo utilities 
GitHub

envgen.py — environment generator (random maps with a specified obstacle coverage) 
GitHub

run_rrt.py — runs the RRT planner on generated environments and reports metrics 
GitHub

optimize_optuna3.py — Optuna optimization script (NSGA-II multi-objective) 
GitHub

Statistical_Significance_Test.py — paired validation + significance testing 
GitHub

example.py — example run / quickstart utility 
GitHub

setup.py — package setup

This repository contains the code used for the project “Black Box Optimization of Planning Parameters in Robotics”. All parameters used in the paper are already hard-coded in the scripts. The instructions below explain which files to run and which parameter values are used for baseline and optimized comparisons.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Required software

Python 3.x

Required Python packages:

numpy

scipy

matplotlib

optuna

If needed, install with:
pip install numpy scipy matplotlib optuna

(Optional) install the project as a package:
pip install -e .

**Files you need to run**

You only need to run two files to reproduce the paper results:

optimize_optuna3.py
Runs NSGA-II optimization using Optuna to find good RRT parameters.

Statistical_Significance_Test.py
Runs paired validation comparing baseline vs optimized parameters and performs statistical testing.

Other files (rrt_algorithms/, envgen.py, run_rrt.py) are helper modules used internally and do not need to be run directly.

**Baseline parameters (fixed)**

The baseline configuration used in ALL baseline runs is:

step_size = 5.0

goal_bias = 0.1

These values are fixed and are used for all obstacle densities.

**Optimized parameters (used in validation/statistical significance)**

The optimized parameters were selected from the Pareto fronts produced by Optuna and are hard-coded (or manually inserted) in Statistical_Significance_Test.py.

Optimized parameters by obstacle density:

10% obstacles:
step_size = 5.298
goal_bias = 0.032

20% obstacles:
step_size = 5.698
goal_bias = 0.058

30% obstacles:
step_size = 4.917
goal_bias = 0.049

40% obstacles:
step_size = 3.818
goal_bias = 0.038

50% obstacles:
step_size = 0.869
goal_bias = 0.054

These are the values reported in Table 1 of the paper and are the values compared against the baseline in the statistical tests.

**Running the optimization**

To reproduce the optimization procedure (Pareto fronts):

Run:
python optimize_optuna3.py

The script uses the following settings internally:

obstacle densities: 10%, 20%, 30%, 40%, 50%

Optuna trials per density: 60

evaluations per trial: 20 random environments

NSGA-II sampler seed: 42

objectives minimized:

execution time

path cost

failure rate

failure penalty:
time = 9999
cost = 9999
failure = 1.0

The optimization output is used to select the optimized parameters listed above.

**Running validation and statistical testing**

To reproduce the validation results and paired t-test:

Run:
python Statistical_Significance_Test.py

This script:

compares baseline vs optimized parameters

uses paired trials (same environment for both planners)

runs 25 paired trials per obstacle density

total paired comparisons: 125

scoring function:
score = 0.3 * (time / 10.0) + 0.7 * ((cost - 140) / 60.0)

failed runs receive score = 1000

statistical test: paired t-test

significance level: alpha = 0.05

random seed = 42

