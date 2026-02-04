import sys
import os
import itertools
import logging
import numpy as np
import optuna
import pandas as pd
import uuid
from DataAnalysis import areaOfContinuousContrast, eventStructuraRatio

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
        "distance_radius_lookup": {"min": 4, "max": 12, "step": 2, "type": "int"},
        "density_multiplier": {"min": 1.2, "max": 3.0, "type": "float"},
        "min_cluster_size": {"min": 5, "max": 100, "log": True, "type": "int"},
        "ratio_ms_to_px": {"min": 15.0, "max": 45.0, "type": "float"},
        "DBSCAN_eps": {"min": 2.0, "max": 10.0, "type": "float"},
        "min_consec": {"min": 1, "max": 5, "type": "int"},
    },
    "DBSCAN.DBSCAN_allEvents": {
        "distance_radius_lookup": {"min": 4, "max": 12, "step": 2, "type": "int"},
        "density_multiplier": {"min": 1.2, "max": 3.0, "type": "float"},
        "min_cluster_size": {"min": 5, "max": 100, "log": True, "type": "int"},
        "ratio_ms_to_px": {"min": 15.0, "max": 45.0, "type": "float"},
        "DBSCAN_eps": {"min": 2.0, "max": 10.0, "type": "float"},
        "padding_xy": {"min": 0, "max": 4, "type": "int"},
        "min_consec": {"min": 1, "max": 5, "type": "int"},
    },
    "DBSCAN.DBSCAN_allEvents_remove_outliers": {
        "distance_radius_lookup": {"min": 4, "max": 12, "step": 2, "type": "int"},
        "density_multiplier": {"min": 1.2, "max": 3.0, "type": "float"},
        "min_cluster_size": {"min": 5, "max": 100, "log": True, "type": "int"},
        "ratio_ms_to_px": {"min": 15.0, "max": 45.0, "type": "float"},
        "DBSCAN_eps": {"min": 2.0, "max": 10.0, "type": "float"},
        "padding_xy": {"min": 0, "max": 4, "type": "int"},
        "outlier_removal_radius": {"min": 2.0, "max": 6.0, "type": "float"},
        "outlier_removal_nbPoints": {"min": 20, "max": 60, "type": "int"},
        "min_consec": {"min": 1, "max": 5, "type": "int"},
    },
}

EIGEN_GRID = {
    "EigenFeatureAnalysis.eigenFeature_analysis": {
        "search_n_neighbours": {"min": 30, "max": 120, "step": 5, "type": "int"},
        "max_eigenval_cutoff": {"min": 0.0, "max": 9.0, "type": "float"},
        "linearity_cutoff": {"min": 0.5, "max": 0.85, "type": "float"},
        "ratio_ms_to_px": {"min": 15.0, "max": 45.0, "type": "float"},
        "DBSCAN_eps": {"min": 2.0, "max": 6.0, "type": "float"},
        "DBSCAN_n_neighbours": {"min": 15, "max": 35, "type": "int"},
    },
    "EigenFeatureAnalysis.eigenFeature_analysis_and_bbox_finding": {
        "search_n_neighbours": {"min": 30, "max": 120, "step": 5, "type": "int"},
        "max_eigenval_cutoff": {"min": 0.0, "max": 9.0, "type": "float"},
        "linearity_cutoff": {"min": 0.5, "max": 0.85, "type": "float"},
        "ratio_ms_to_px": {"min": 15.0, "max": 45.0, "type": "float"},
        "DBSCAN_eps": {"min": 2.0, "max": 6.0, "type": "float"},
        "DBSCAN_n_neighbours": {"min": 15, "max": 35, "type": "int"},
        "bbox_padding": {"min": 0, "max": 3, "type": "int"},
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

def optimize_method_with_optuna(method_name, param_grid, filtered_events, settings, n_trials=200, scoring_method='count', storage_url=None, study_name=None):

    """
    Use Optuna to optimize parameters for a single method using Bayesian optimization.

    Args:
        method_name: Name of the method (e.g., "DBSCAN.DBSCAN_onlyHighDensity")
        param_grid: Dictionary of parameter ranges
        filtered_events: Filtered event data
        settings: Global settings
        n_trials: Number of Optuna trials to run (default 200)
        scoring_method: Method to score the run ('count', 'areaOfContinuousContrast', 'eventStructuraRatio')

    Returns:
        (best_score, best_params, method_name)
    """
    func = get_function_from_name(method_name)
    if not func:
        logging.warning(f"Could not find function for {method_name}")
        return -1, {}, method_name

    def objective(trial):
        kwargs = {}
        # Ensure debug is set to avoid KeyErrors in some finding methods
        kwargs['debug'] = 'False'
        
        for param_name, param_values in param_grid.items():
            if isinstance(param_values, dict) and 'min' in param_values:
                # Handle range definition
                low = param_values['min']
                high = param_values['max']
                step = param_values.get('step', None)
                log = param_values.get('log', False)
                p_type = param_values.get('type', 'float')

                if p_type == 'int':
                    # suggest_int requires step to be a valid integer if provided, defaulting to 1
                    effective_step = step if step is not None else 1
                    # If log=True, step is usually ignored or must be 1 (depending on Optuna version), enforce 1 to be safe
                    if log:
                        effective_step = 1
                    kwargs[param_name] = trial.suggest_int(param_name, low, high, step=effective_step, log=log)
                else:
                    kwargs[param_name] = trial.suggest_float(param_name, low, high, step=step, log=log)
            elif isinstance(param_values, list):
                # Backwards compatibility for lists (e.g., FRAME_GRID)
                if all(isinstance(v, (int, np.integer)) for v in param_values):
                    # Integer parameter
                    if len(set(param_values)) <= 10:
                        kwargs[param_name] = trial.suggest_categorical(param_name, param_values)
                    else:
                        min_val, max_val = min(param_values), max(param_values)
                        sorted_vals = sorted(set(param_values))
                        step = sorted_vals[1] - sorted_vals[0] if len(sorted_vals) > 1 else 1
                        kwargs[param_name] = trial.suggest_int(param_name, min_val, max_val, step=step)
                elif all(isinstance(v, (float, np.floating)) for v in param_values):
                    # Float parameter
                    if len(set(param_values)) <= 10:
                        kwargs[param_name] = trial.suggest_categorical(param_name, param_values)
                    else:
                        kwargs[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                else:
                    kwargs[param_name] = trial.suggest_categorical(param_name, param_values)

        try:
            candidates, _ = func(filtered_events, settings, **kwargs)
            
            if scoring_method == 'count':
                score = len(candidates) if candidates is not None else 0
            else:
                if not candidates or len(candidates) == 0:
                    score = 0
                else:
                    # Collect all events from candidates
                    all_dfs = [c['events'] for c in candidates.values() if 'events' in c]
                    if not all_dfs:
                        score = 0
                    else:
                        combined_df = pd.concat(all_dfs)
                        # Analysis scripts expect a record array or similar that supports ev['x']
                        ev_rec = combined_df.to_records(index=False)
                        
                        # Default resolutions
                        x_res = 941
                        y_res = 483
                        
                        if scoring_method == 'areaOfContinuousContrast':
                            _, results = areaOfContinuousContrast.run_analysis(ev_rec, x_res=x_res, y_res=y_res)
                            score = results.get('area', 0)
                        elif scoring_method == 'eventStructuraRatio':
                            _, results = eventStructuraRatio.run_analysis(ev_rec, x_res, y_res)
                            score = results.get('score', 0)
                        else:
                            score = len(candidates)
        except Exception as e:
            logging.exception(f"Trial failed for {method_name}: {e}")
            score = -1

        return score

    # storage_url = "sqlite:///db.sqlite3"
    # # Create study with TPE sampler for Bayesian optimization
    # study = optuna.create_study(
    #     direction="maximize",
    #     sampler=optuna.samplers.TPESampler(seed=42),
    #     storage=storage_url
    # )
    
    if storage_url is not None and study_name is not None:
        study = optuna.load_study(
            study_name=study_name, 
            storage=storage_url,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
    else:
        # Fallback for standalone usage or if arguments not provided
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            storage="sqlite:///db.sqlite3",
            pruner=optuna.pruners.MedianPruner()
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
        keys = []
        values_lists = []
        for p, v in param_grid.items():
            keys.append(p)
            if isinstance(v, dict) and 'min' in v:
                # Create a simple grid for preview
                # We interpret the range by taking 3 samples (min, mid, max) to keep it fast
                if v.get('type') == 'int':
                    steps = 3
                    if v.get('max') - v.get('min') < 3:
                        steps = v.get('max') - v.get('min') + 1
                    vals = np.linspace(v['min'], v['max'], int(steps))
                    vals = np.unique(np.round(vals).astype(int))
                else:
                    vals = np.linspace(v['min'], v['max'], 3)
                values_lists.append(vals.tolist())
            else:
                values_lists.append(v)
        
        combinations = list(itertools.product(*values_lists))
        
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

def search_run_optuna(npy_array, settings, time_stretch=None, xy_stretch=None, n_trials=200, n_jobs=1, scoring_method='count'):
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
        scoring_method: Method to score the run ('count', 'areaOfContinuousContrast', 'eventStructuraRatio')

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

    all_grids = {**DBSCAN_GRID, **EIGEN_GRID}

    storage_url = "sqlite:///db.sqlite3"
    study_names = {}
    
    # Pre-create studies sequentially to avoid SQLite locking issues in parallel execution
    print("Initializing Optuna studies...")
    for m_name in all_grids.keys():
        unique_name = f"{m_name}_{uuid.uuid4().hex[:8]}"
        optuna.create_study(
            study_name=unique_name,
            storage=storage_url,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        study_names[m_name] = unique_name

    if n_jobs != 1:
        print(f"Optimizing {len(all_grids)} methods in parallel with n_jobs={n_jobs}...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(optimize_method_with_optuna)(
                method_name, param_grid, filtered_events, settings, n_trials, scoring_method, storage_url, study_names[method_name]
            )
            for method_name, param_grid in all_grids.items()
        )
    else:
        # Run sequentially with progress updates
        results = []
        for i, (method_name, param_grid) in enumerate(all_grids.items(), 1):
            print(f"[{i}/{len(all_grids)}] Optimizing {method_name}...")
            score, params, method = optimize_method_with_optuna(
                method_name, param_grid, filtered_events, settings, n_trials, scoring_method, storage_url, study_names[method_name]
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
