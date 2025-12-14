#!/usr/bin/env python3
"""
RRT-Specific Benchmark Suite
Comprehensive evaluation of optimized vs baseline RRT parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from run_rrt import run_rrt
from envgen import generate_random_obstacles_pct

# Your specific parameters
OPTIMIZED_PARAMS = {
    "step_size": 1.78,
    "goal_bias": 0.098
}

BASELINE_PARAMS = {
    "step_size": 10.0,
    "goal_bias": 0.25
}

def calculate_performance_score(exec_time, path_cost, success, time_weight=0.3, cost_weight=0.7):
    """
    Calculate a combined performance score
    Lower score is better
    """
    if not success:
        return 1000.0  # Heavy penalty for failure
    
    # Normalize time (assuming typical range 0-10 seconds)
    normalized_time = min(exec_time / 10.0, 1.0)
    
    # Normalize cost (assuming typical range 140-200 units)  
    normalized_cost = min((path_cost - 140) / 60.0, 1.0)
    
    # Combined score
    score = (time_weight * normalized_time) + (cost_weight * normalized_cost)
    return score

def create_benchmark_suite():
    """Create diverse path planning benchmark problems"""
    benchmarks = []
    
    # 1. Level 1 obstacles (10% coverage)
    benchmarks.append({
        'name': 'Level1_10pct',
        'coverage_range': (8, 12),
        'complexity': 'level1'
    })
    
    # 2. Level 2 obstacles (20% coverage)  
    benchmarks.append({
        'name': 'Level2_20pct', 
        'coverage_range': (18, 22),
        'complexity': 'level2'
    })
    
    # 3. :evel 3 obstacles (30% coverage)
    benchmarks.append({
        'name': 'Level3_30pct',
        'coverage_range': (28, 32),
        'complexity': 'level3'
    })
    
    # 4. Level 4 obstacles (40% coverage)
    benchmarks.append({
        'name': 'Level4_40pct',
        'coverage_range': (38, 42),
        'complexity': 'level4'
    })

    # 4. Level 5 obstacles (50% coverage)
    benchmarks.append({
        'name': 'Level5_50pct',
        'coverage_range': (48, 52),
        'complexity': 'level5'
    })
    
    
    return benchmarks

def run_single_rrt_evaluation(obstacles, params):
    """Run RRT with given parameters and return performance metrics"""
    X_dimensions = np.array([(0, 100), (0, 100)])
    start = (0, 0)
    goal = (100, 100)
    
    try:
        exec_time, path_cost, samples = run_rrt(
            obstacles, start, goal,
            step_size=params["step_size"],
            goal_bias=params["goal_bias"]
        )
        
        success = (exec_time is not None) and (path_cost is not None)
        if success:
            score = calculate_performance_score(exec_time, path_cost, success)
            return {
                'success': True,
                'exec_time': exec_time,
                'path_cost': path_cost,
                'performance_score': score,
                'samples': samples
            }
        else:
            return {
                'success': False,
                'exec_time': None,
                'path_cost': None,
                'performance_score': 1000.0,
                'samples': 0
            }
            
    except Exception as e:
        return {
            'success': False,
            'exec_time': None,
            'path_cost': None,
            'performance_score': 1000.0,
            'samples': 0,
            'error': str(e)
        }

def benchmark_rrt_parameters(optimized_params, baseline_params, n_trials_per_benchmark=25):
    """Comprehensive benchmarking across diverse scenarios"""
    benchmarks = create_benchmark_suite()
    results = {}
    
    for benchmark in benchmarks:
        print(f"Testing {benchmark['name']}...")
        
        benchmark_results = []
        for trial in range(n_trials_per_benchmark):
            # Create reproducible test environment
            np.random.seed(trial * 1000 + hash(benchmark['name']) % 1000)
            
            coverage = np.random.uniform(*benchmark['coverage_range'])
            X_dimensions = np.array([(0, 100), (0, 100)])
            start = (0, 0)
            goal = (100, 100)
            
            obstacles = generate_random_obstacles_pct(
                X_dimensions, start, goal, coverage_percentage=coverage
            )
            
            # Test optimized parameters
            opt_result = run_single_rrt_evaluation(obstacles, optimized_params)
            
            # Test baseline parameters on SAME environment
            base_result = run_single_rrt_evaluation(obstacles, baseline_params)
            
            benchmark_results.append({
                'trial_id': trial,
                'coverage': coverage,
                'optimized_success': opt_result['success'],
                'baseline_success': base_result['success'],
                'optimized_time': opt_result['exec_time'],
                'baseline_time': base_result['exec_time'],
                'optimized_cost': opt_result['path_cost'],
                'baseline_cost': base_result['path_cost'],
                'optimized_score': opt_result['performance_score'],
                'baseline_score': base_result['performance_score'],
                'cost_improvement': ((base_result['path_cost'] - opt_result['path_cost']) / base_result['path_cost']) * 100 
                    if opt_result['success'] and base_result['success'] else 0,
                'time_improvement': ((base_result['exec_time'] - opt_result['exec_time']) / base_result['exec_time']) * 100 
                    if opt_result['success'] and base_result['success'] else 0,
                'score_improvement': ((base_result['performance_score'] - opt_result['performance_score']) / base_result['performance_score']) * 100
            })
        
        results[benchmark['name']] = benchmark_results
    
    return results, benchmarks

def statistical_significance_test(results):
    """Formal statistical testing of performance differences"""
    print("\n" + "="*60)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*60)
    
    all_optimized_scores = []
    all_baseline_scores = []
    
    for benchmark_name, benchmark_results in results.items():
        optimized_scores = [r['optimized_score'] for r in benchmark_results]
        baseline_scores = [r['baseline_score'] for r in benchmark_results]
        
        all_optimized_scores.extend(optimized_scores)
        all_baseline_scores.extend(baseline_scores)
        
        # Individual benchmark test
        t_stat, p_value = stats.ttest_rel(optimized_scores, baseline_scores)
        
        print(f"\n{benchmark_name}:")
        print(f"  T-statistic: {t_stat:.3f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Significant improvement (p<0.05): {p_value < 0.05}")
        print(f"  Mean optimized score: {np.mean(optimized_scores):.3f}")
        print(f"  Mean baseline score: {np.mean(baseline_scores):.3f}")
    
    # Overall test
    overall_t, overall_p = stats.ttest_rel(all_optimized_scores, all_baseline_scores)
    print(f"\nOVERALL ACROSS ALL BENCHMARKS:")
    print(f"  T-statistic: {overall_t:.3f}")
    print(f"  P-value: {overall_p:.10f}")
    print(f"  Significant improvement (p<0.05): {overall_p < 0.05}")
    print(f"  Mean optimized score: {np.mean(all_optimized_scores):.3f}")
    print(f"  Mean baseline score: {np.mean(all_baseline_scores):.3f}")
    
    return overall_t, overall_p

def create_comprehensive_plots(results, benchmarks):
    """Create eval plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RRT Parameter Optimization: Optimized vs Baseline Performance', fontsize=16, fontweight='bold')
    
    # Prepare data
    benchmark_names = [b['name'] for b in benchmarks]
    
    # 1. Success Rate Comparison
    success_rates_opt = []
    success_rates_base = []
    
    for benchmark_name in benchmark_names:
        benchmark_results = results[benchmark_name]
        opt_success = np.mean([r['optimized_success'] for r in benchmark_results]) * 100
        base_success = np.mean([r['baseline_success'] for r in benchmark_results]) * 100
        success_rates_opt.append(opt_success)
        success_rates_base.append(base_success)
    
    x_pos = np.arange(len(benchmark_names))
    width = 0.35
    
    axes[0, 0].bar(x_pos - width/2, success_rates_base, width, label='Baseline', alpha=0.7, color='red')
    axes[0, 0].bar(x_pos + width/2, success_rates_opt, width, label='Optimized', alpha=0.7, color='green')
    axes[0, 0].set_xlabel('Benchmark Scenario')
    axes[0, 0].set_ylabel('Success Rate (%)')
    axes[0, 0].set_title('Success Rate Comparison')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels([name.split('_')[0] for name in benchmark_names], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Performance Score Comparison
    mean_scores_opt = [np.mean([r['optimized_score'] for r in results[name]]) for name in benchmark_names]
    mean_scores_base = [np.mean([r['baseline_score'] for r in results[name]]) for name in benchmark_names]
    
    axes[0, 1].bar(x_pos - width/2, mean_scores_base, width, label='Baseline', alpha=0.7, color='red')
    axes[0, 1].bar(x_pos + width/2, mean_scores_opt, width, label='Optimized', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Benchmark Scenario')
    axes[0, 1].set_ylabel('Performance Score (lower is better)')
    axes[0, 1].set_title('Performance Score Comparison')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([name.split('_')[0] for name in benchmark_names], rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Path Cost Improvement
    cost_improvements = []
    for benchmark_name in benchmark_names:
        improvements = [r['cost_improvement'] for r in results[benchmark_name] if r['optimized_success'] and r['baseline_success']]
        cost_improvements.append(np.mean(improvements) if improvements else 0)
    
    colors = ['green' if x > 0 else 'red' for x in cost_improvements]
    bars = axes[0, 2].bar(benchmark_names, cost_improvements, color=colors, alpha=0.7)
    axes[0, 2].set_xlabel('Benchmark Scenario')
    axes[0, 2].set_ylabel('Path Cost Improvement (%)')
    axes[0, 2].set_title('Path Cost Improvement\n(positive = optimized better)')
    axes[0, 2].set_xticklabels([name.split('_')[0] for name in benchmark_names], rotation=45)
    axes[0, 2].axhline(0, color='black', linewidth=0.8)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Value labels on bars
    for bar, imp in zip(bars, cost_improvements):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if imp > 0 else -3),
                       f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top', fontweight='bold')
    
    # 4. Execution Time Improvement
    time_improvements = []
    for benchmark_name in benchmark_names:
        improvements = [r['time_improvement'] for r in results[benchmark_name] if r['optimized_success'] and r['baseline_success']]
        time_improvements.append(np.mean(improvements) if improvements else 0)
    
    colors = ['green' if x > 0 else 'red' for x in time_improvements]
    bars = axes[1, 0].bar(benchmark_names, time_improvements, color=colors, alpha=0.7)
    axes[1, 0].set_xlabel('Benchmark Scenario')
    axes[1, 0].set_ylabel('Execution Time Improvement (%)')
    axes[1, 0].set_title('Execution Time Improvement\n(positive = optimized better)')
    axes[1, 0].set_xticklabels([name.split('_')[0] for name in benchmark_names], rotation=45)
    axes[1, 0].axhline(0, color='black', linewidth=0.8)
    axes[1, 0].grid(True, alpha=0.3)
    
    for bar, imp in zip(bars, time_improvements):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if imp > 0 else -3),
                       f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top', fontweight='bold')
    
    # 5. Score Improvement Distribution
    all_score_improvements = []
    for benchmark_name in benchmark_names:
        improvements = [r['score_improvement'] for r in results[benchmark_name]]
        all_score_improvements.extend(improvements)
    
    axes[1, 1].hist(all_score_improvements, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].axvline(np.mean(all_score_improvements), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(all_score_improvements):.1f}%')
    axes[1, 1].axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[1, 1].set_xlabel('Performance Score Improvement (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Performance Improvements')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Summary Stats
    axes[1, 2].axis('off')
    summary_text = (
        f"Optimized Parameters:\n"
        f"  step_size: {OPTIMIZED_PARAMS['step_size']}\n"
        f"  goal_bias: {OPTIMIZED_PARAMS['goal_bias']}\n\n"
        f"Baseline Parameters:\n"
        f"  step_size: {BASELINE_PARAMS['step_size']}\n"
        f"  goal_bias: {BASELINE_PARAMS['goal_bias']}\n\n"
        f"Overall Results:\n"
        f"  Total trials: {sum(len(results[name]) for name in benchmark_names)}\n"
        f"  Avg success rate (optimized): {np.mean(success_rates_opt):.1f}%\n"
        f"  Avg success rate (baseline): {np.mean(success_rates_base):.1f}%\n"
        f"  Avg performance improvement: {np.mean(all_score_improvements):.1f}%"
    )
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, fontfamily='monospace',
                   verticalalignment='top', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig('rrt_optimization_benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_report(results, benchmarks):
    """Print comprehensive performance report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE RRT PARAMETER OPTIMIZATION REPORT")
    print("="*80)
    
    print(f"\nPARAMETERS:")
    print(f"  Optimized: step_size={OPTIMIZED_PARAMS['step_size']:.3f}, goal_bias={OPTIMIZED_PARAMS['goal_bias']:.3f}")
    print(f"  Baseline:  step_size={BASELINE_PARAMS['step_size']:.3f}, goal_bias={BASELINE_PARAMS['goal_bias']:.3f}")
    
    for benchmark in benchmarks:
        benchmark_results = results[benchmark['name']]
        
        opt_success_rate = np.mean([r['optimized_success'] for r in benchmark_results]) * 100
        base_success_rate = np.mean([r['baseline_success'] for r in benchmark_results]) * 100
        
        opt_scores = [r['optimized_score'] for r in benchmark_results]
        base_scores = [r['baseline_score'] for r in benchmark_results]
        
        # Only consider successful runs for cost/time comparisons
        successful_runs = [r for r in benchmark_results if r['optimized_success'] and r['baseline_success']]
        
        if successful_runs:
            avg_cost_improvement = np.mean([r['cost_improvement'] for r in successful_runs])
            avg_time_improvement = np.mean([r['time_improvement'] for r in successful_runs])
        else:
            avg_cost_improvement = avg_time_improvement = 0
        
        print(f"\n{benchmark['name']} ({benchmark['coverage_range'][0]}-{benchmark['coverage_range'][1]}% coverage):")
        print(f"  Success Rate: {opt_success_rate:.1f}% (optimized) vs {base_success_rate:.1f}% (baseline)")
        print(f"  Performance Score: {np.mean(opt_scores):.3f} vs {np.mean(base_scores):.3f}")
        if successful_runs:
            print(f"  Path Cost Improvement: {avg_cost_improvement:+.1f}%")
            print(f"  Execution Time Improvement: {avg_time_improvement:+.1f}%")

if __name__ == "__main__":
    print("RRT Parameter Optimization Benchmarking")
    print("="*50)
    
    # Run comprehensive benchmarking
    results, benchmarks = benchmark_rrt_parameters(OPTIMIZED_PARAMS, BASELINE_PARAMS, n_trials_per_benchmark=25)
    
    # Statistical significance testing
    statistical_significance_test(results)
    
    # Create visualizations
    create_comprehensive_plots(results, benchmarks)
    
    # Print detailed report
    print_detailed_report(results, benchmarks)
    

    print(f"\nBenchmarking completed! Results saved to 'rrt_optimization_benchmark_results.png'")
