"""
Example script demonstrating how to use Optuna-based Bayesian optimization
for hyperparameter search in EVE's candidate finding algorithms.

This script shows two approaches:
1. Sequential optimization (recommended for most cases)
2. Parallel optimization across methods (for faster results)
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HyperParameterSearch.findingSearch import preview_run_optuna

# Example: Load your event data
# npy_array should be a structured numpy array with fields: x, y, p, t
# Example format:
# npy_array = np.load('your_event_data.npy')

# Example settings dictionary (adjust based on your EVE configuration)
settings = {
    'max_parallel': 8,  # Limit parallel processes
    # Add other EVE settings as needed
}

# Example 1: Sequential optimization (default, shows progress)
def example_sequential(npy_array, settings):
    """
    Run Optuna optimization sequentially through all methods.
    This is recommended for most use cases as it provides clear progress updates.
    """
    print("Example 1: Sequential Optuna Optimization")
    print("-" * 60)
    
    best_method, best_params = preview_run_optuna(
        npy_array=npy_array,
        settings=settings,
        time_stretch=None,      # Optional: (start_ms, duration_ms)
        xy_stretch=None,        # Optional: (min_x, max_x, min_y, max_y)
        n_trials=200,           # Number of trials per method (default: 200)
        n_jobs=1                # Sequential processing
    )
    
    return best_method, best_params


# Example 2: Parallel optimization across methods
def example_parallel(npy_array, settings):
    """
    Run Optuna optimization in parallel across different methods.
    This is faster but provides less visibility into progress.
    """
    print("Example 2: Parallel Optuna Optimization")
    print("-" * 60)
    
    best_method, best_params = preview_run_optuna(
        npy_array=npy_array,
        settings=settings,
        time_stretch=None,
        xy_stretch=None,
        n_trials=200,
        n_jobs=4                # Use 4 parallel jobs (adjust based on CPU cores)
    )
    
    return best_method, best_params


# Example 3: Optimization on a subset of data
def example_subset(npy_array, settings):
    """
    Run Optuna optimization on a temporal and spatial subset of the data.
    This is useful for faster testing on representative data.
    """
    print("Example 3: Optuna Optimization on Data Subset")
    print("-" * 60)
    
    best_method, best_params = preview_run_optuna(
        npy_array=npy_array,
        settings=settings,
        time_stretch=(0, 1000),         # First 1000ms of data
        xy_stretch=(0, 100, 0, 100),    # 100x100 pixel region
        n_trials=100,                    # Fewer trials for quick testing
        n_jobs=1
    )
    
    return best_method, best_params


# Example 4: Quick test with fewer trials
def example_quick_test(npy_array, settings):
    """
    Quick optimization with fewer trials for rapid prototyping.
    """
    print("Example 4: Quick Optuna Test")
    print("-" * 60)
    
    best_method, best_params = preview_run_optuna(
        npy_array=npy_array,
        settings=settings,
        n_trials=50,            # Fewer trials for quick test
        n_jobs=1
    )
    
    return best_method, best_params


if __name__ == "__main__":
    # Example usage:
    # 
    # # Load your event data
    # npy_array = np.load('path/to/your/event_data.npy')
    # 
    # # Run optimization
    # best_method, best_params = example_sequential(npy_array, settings)
    # 
    # # Use the best parameters for actual candidate finding
    # from HyperParameterSearch.findingSearch import get_function_from_name
    # func = get_function_from_name(best_method)
    # candidates, _ = func(npy_array, settings, **best_params)
    # 
    # print(f"Found {len(candidates)} candidates using optimized parameters!")
    
    print("Example script - replace with actual data loading and usage")
    print("\nKey advantages of Optuna (Bayesian optimization) over grid search:")
    print("1. More efficient: Explores parameter space intelligently")
    print("2. Faster: Typically finds good parameters with fewer evaluations")
    print("3. Adaptive: Learns from previous trials to focus on promising regions")
    print("4. Handles continuous parameters better than discrete grid search")
    print("\nTo use: Load your event data and call preview_run_optuna()")

