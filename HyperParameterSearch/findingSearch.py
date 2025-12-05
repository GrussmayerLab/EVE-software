import sys
import os
import itertools
import logging
import numpy as np
import optuna

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from eve_smlm.CandidateFinding import DBSCAN, EigenFeatureAnalysis, FrameBasedFinding
except ImportError:
    from CandidateFinding import DBSCAN, EigenFeatureAnalysis, FrameBasedFinding

RATIO_RANGE = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
DBSCAN_EPS_RANGE = [4, 5, 6, 7, 8, 9, 10]

DBSCAN_GRID = {
    "DBSCAN.DBSCAN_onlyHighDensity": {
        "distance_radius_lookup": [4, 6, 8, 10, 12],
        "density_multiplier": [1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0],
        "min_cluster_size": [10, 15, 20, 25, 30, 35, 40],
        "ratio_ms_to_px": RATIO_RANGE,
        "DBSCAN_eps": DBSCAN_EPS_RANGE,
        "min_consec": [1, 2, 3, 4, 5],
    },
    "DBSCAN.DBSCAN_allEvents": {
        "distance_radius_lookup": [4, 6, 8, 10, 12],
        "density_multiplier": [1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0],
        "min_cluster_size": [10, 15, 20, 25, 30, 35, 40],
        "ratio_ms_to_px": RATIO_RANGE,
        "DBSCAN_eps": DBSCAN_EPS_RANGE,
        "padding_xy": [0, 1, 2, 3, 4],
        "min_consec": [1, 2, 3, 4, 5],
    },
    "DBSCAN.DBSCAN_allEvents_remove_outliers": {
        "distance_radius_lookup": [4, 6, 8, 10, 12],
        "density_multiplier": [1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0],
        "min_cluster_size": [10, 15, 20, 25, 30, 35, 40],
        "ratio_ms_to_px": RATIO_RANGE,
        "DBSCAN_eps": DBSCAN_EPS_RANGE,
        "padding_xy": [0, 1, 2, 3, 4],
        "outlier_removal_radius": [2, 3, 4, 5, 6],
        "outlier_removal_nbPoints": [20, 30, 40, 50, 60],
        "min_consec": [1, 2, 3, 4, 5],
    },
}

EIGEN_GRID = {
    "EigenFeatureAnalysis.eigenFeature_analysis": {
        "search_n_neighbours": [30, 45, 60, 75, 90, 105, 120],
        "max_eigenval_cutoff": [0.0, 3.0, 5.0, 7.0, 9.0],
        "linearity_cutoff": [0.5, 0.6, 0.7, 0.8, 0.85],
        "ratio_ms_to_px": [15.0, 20.0, 25.0, 30.0, 35.0],
        "DBSCAN_eps": [2, 3, 4, 5, 6],
        "DBSCAN_n_neighbours": [15, 20, 25, 30, 35],
    },
    "EigenFeatureAnalysis.eigenFeature_analysis_and_bbox_finding": {
        "search_n_neighbours": [30, 45, 60, 75, 90, 105, 120],
        "max_eigenval_cutoff": [0.0, 3.0, 5.0, 7.0, 9.0],
        "linearity_cutoff": [0.5, 0.6, 0.7, 0.8, 0.85],
        "ratio_ms_to_px": [15.0, 20.0, 25.0, 30.0, 35.0],
        "DBSCAN_eps": [2, 3, 4, 5, 6],
        "DBSCAN_n_neighbours": [15, 20, 25, 30, 35],
        "bbox_padding": [0, 1, 2, 3],
    },
}

FRAME_GRID = {
    "FrameBasedFinding.FrameBased_finding": {
        "threshold_detection": [2.5, 3.0, 3.5, 4.0, 4.5],
        "exclusion_radius": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "min_diameter": [1.0, 1.25, 1.5, 1.75, 2.0],
        "max_diameter": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        "frame_time": [50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0],
        "candidate_radius": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    }
}

def get_function_from_name(name):
    module_name, func_name = name.split('.')
    if module_name == 'DBSCAN':
        return getattr(DBSCAN, func_name)
    elif module_name == 'EigenFeatureAnalysis':
        return getattr(EigenFeatureAnalysis, func_name)
    elif module_name == 'FrameBasedFinding':
        return getattr(FrameBasedFinding, func_name)
    return None

def filter_events_t(events, t_stretch):
    """
    Filter events to a certain time-stretch.
    t_stretch is (start_ms, duration_ms)
    """
    if t_stretch is None:
        return events
    
    start_ms, duration_ms = t_stretch
    # events['t'] is typically in microseconds
    t_min = float(start_ms) * 1000
    t_max = t_min + float(duration_ms) * 1000
    
    indices = np.where((events['t'] >= t_min) & (events['t'] <= t_max))
    return events[indices]

def filter_events_xy(events, xy_stretch):
    """
    Filter events to a certain xy-stretch.
    xy_stretch is (min_x, max_x, min_y, max_y)
    """
    if xy_stretch is None:
        return events

    try:
        min_x, max_x, min_y, max_y = map(float, xy_stretch)
        
        # Check for valid infinite bounds or positive values
        if (min_x > 0) or (min_y > 0) or (max_x < np.inf) or (max_y < np.inf):
             mask = (events['x'] >= min_x) & (events['x'] <= max_x) & \
                    (events['y'] >= min_y) & (events['y'] <= max_y)
             return events[mask]
    except Exception as e:
        logging.warning(f"XY filtering failed: {e}")
        
    return events

from joblib import Parallel, delayed

def evaluate_params(method_name, func, kwargs, events, settings):
    """
    Helper function to evaluate a single set of parameters.
    Returns: (score, kwargs, method_name)
    """
    try:
        candidates, _ = func(events, settings, **kwargs)
        score = len(candidates) if candidates is not None else 0
        return score, kwargs, method_name
    except Exception as e:
        logging.error(f"Error running {method_name} with {kwargs}: {e}")
        return -1, kwargs, method_name

def optimize_method_with_optuna(method_name, param_grid, filtered_events, settings, n_trials=200):
    """
    Use Optuna to optimize parameters for a single method using Bayesian optimization.

    Args:
        method_name: Name of the method (e.g., "DBSCAN.DBSCAN_onlyHighDensity")
        param_grid: Dictionary of parameter ranges
        filtered_events: Filtered event data
        settings: Global settings
        n_trials: Number of Optuna trials to run (default 200)

    Returns:
        (best_score, best_params, method_name)
    """
    func = get_function_from_name(method_name)
    if not func:
        logging.warning(f"Could not find function for {method_name}")
        return -1, {}, method_name

    def objective(trial):
        kwargs = {}
        for param_name, param_values in param_grid.items():
            if isinstance(param_values, list):
                # Determine the type of parameter
                if all(isinstance(v, (int, np.integer)) for v in param_values):
                    # Integer parameter
                    if len(set(param_values)) <= 10:
                        # Use categorical for small discrete sets
                        kwargs[param_name] = trial.suggest_categorical(param_name, param_values)
                    else:
                        # Use int range for larger sets
                        min_val, max_val = min(param_values), max(param_values)
                        # Try to detect step
                        sorted_vals = sorted(set(param_values))
                        step = sorted_vals[1] - sorted_vals[0] if len(sorted_vals) > 1 else 1
                        kwargs[param_name] = trial.suggest_int(param_name, min_val, max_val, step=step)
                elif all(isinstance(v, (float, np.floating)) for v in param_values):
                    # Float parameter
                    if len(set(param_values)) <= 10:
                        # Use categorical for small discrete sets
                        kwargs[param_name] = trial.suggest_categorical(param_name, param_values)
                    else:
                        # Use float range for larger sets
                        kwargs[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                else:
                    # Mixed types or other - use categorical
                    kwargs[param_name] = trial.suggest_categorical(param_name, param_values)

        try:
            candidates, _ = func(filtered_events, settings, **kwargs)
            score = len(candidates) if candidates is not None else 0
        except Exception as e:
            logging.debug(f"Trial failed for {method_name}: {e}")
            score = -1

        return score

    # Create study with TPE sampler for Bayesian optimization
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Suppress Optuna's verbose output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_value, study.best_params, method_name


def preview_run(npy_array, settings, time_stretch=None, xy_stretch=None):
    """
    Runs the finding search using the defined grids on the provided data.
    
    Args:
        npy_array: The event data (numpy record array with x, y, p, t).
        settings: Global settings dictionary.
        time_stretch: Tuple (start_ms, duration_ms) or None.
        xy_stretch: Tuple (min_x, max_x, min_y, max_y) or None.
    """
    # 1. Filter Data (Subset)
    filtered_events = filter_events_t(npy_array, time_stretch)
    filtered_events = filter_events_xy(filtered_events, xy_stretch)
    
    if len(filtered_events) == 0:
        logging.warning("No events found in the specified subset. Aborting search.")
        return None, None

    print(f"Running search on {len(filtered_events)} events (Subset of {len(npy_array)})...")

    all_grids = {**DBSCAN_GRID}
    
    tasks = []

    print(f"Preparing tasks for {len(all_grids)} methods...")

    for method_name, param_grid in all_grids.items():
        func = get_function_from_name(method_name)
        if not func:
            logging.warning(f"Could not find function for {method_name}")
            continue
            
        # Generate parameter combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        for combination in combinations:
            kwargs = dict(zip(keys, combination))
            tasks.append((method_name, func, kwargs))

    print(f"Running {len(tasks)} tasks in parallel...")
    
    # Run in parallel
    # n_jobs=-1 uses all available cores
    results = Parallel(n_jobs=8)(
        delayed(evaluate_params)(method_name, func, kwargs, filtered_events, settings)
        for method_name, func, kwargs in tasks
    )
    
    # Find best result
    best_score = -1
    best_params = None
    best_method = None
    
    for score, params, method in results:
        if score > best_score:
            best_score = score
            best_params = params
            best_method = method

    print("\nSearch Complete.")
    print(f"Best Method: {best_method}")
    print(f"Best Params: {best_params}")
    print(f"Best Score: {best_score}")
    
    return best_method, best_params

def preview_run_optuna(npy_array, settings, time_stretch=None, xy_stretch=None, n_trials=200, n_jobs=1):
    """
    Runs hyperparameter search using Optuna's Bayesian optimization (TPE sampler).

    Args:
        npy_array: The event data (numpy record array with x, y, p, t).
        settings: Global settings dictionary.
        time_stretch: Tuple (start_ms, duration_ms) or None.
        xy_stretch: Tuple (min_x, max_x, min_y, max_y) or None.
        n_trials: Number of trials per method (default 200).
        n_jobs: Number of parallel jobs for running multiple methods (default 1).
                Set to -1 to use all cores, or a specific number to limit.

    Returns:
        (best_method, best_params): The best method name and its optimized parameters.
    """
    # 1. Filter Data (Subset)
    filtered_events = filter_events_t(npy_array, time_stretch)
    filtered_events = filter_events_xy(filtered_events, xy_stretch)

    if len(filtered_events) == 0:
        logging.warning("No events found in the specified subset. Aborting search.")
        return None, None

    print(f"Running Optuna Bayesian optimization on {len(filtered_events)} events...")
    print(f"Using {n_trials} trials per method\n")

    all_grids = {**DBSCAN_GRID, **EIGEN_GRID, **FRAME_GRID}

    # Optionally run methods in parallel (each method gets its own Optuna study)
    if n_jobs != 1:
        print(f"Optimizing {len(all_grids)} methods in parallel with n_jobs={n_jobs}...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(optimize_method_with_optuna)(
                method_name, param_grid, filtered_events, settings, n_trials
            )
            for method_name, param_grid in all_grids.items()
        )
    else:
        # Run sequentially with progress updates
        results = []
        for i, (method_name, param_grid) in enumerate(all_grids.items(), 1):
            print(f"[{i}/{len(all_grids)}] Optimizing {method_name}...")
            score, params, method = optimize_method_with_optuna(
                method_name, param_grid, filtered_events, settings, n_trials
            )
            results.append((score, params, method))
            print(f"      Best score: {score} with {len(params)} parameters")

    # Find best result across all methods
    best_overall_score = -1
    best_overall_params = None
    best_overall_method = None

    for score, params, method in results:
        if score > best_overall_score:
            best_overall_score = score
            best_overall_params = params
            best_overall_method = method

    print("\n" + "="*60)
    print("Optuna Bayesian Optimization Complete!")
    print("="*60)
    print(f"Best Method: {best_overall_method}")
    print(f"Best Score: {best_overall_score}")
    print(f"Best Params:")
    for k, v in best_overall_params.items():
        print(f"  {k}: {v}")
    print("="*60)

    return best_overall_method, best_overall_params

if __name__ == "__main__":
    # Placeholder for running directly if needed
    pass
