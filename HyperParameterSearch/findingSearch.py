import sys
import os
import itertools
import logging
import numpy as np

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

    all_grids = {**DBSCAN_GRID, **EIGEN_GRID, **FRAME_GRID}
    
    best_score = -1
    best_params = None
    best_method = None

    print(f"Starting preview run with {len(all_grids)} methods...")

    for method_name, param_grid in all_grids.items():
        func = get_function_from_name(method_name)
        if not func:
            logging.warning(f"Could not find function for {method_name}")
            continue
            
        print(f"Testing method: {method_name}")
        
        # Generate parameter combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        print(f"  Testing {len(combinations)} combinations...")
        
        for i, combination in enumerate(combinations):
            kwargs = dict(zip(keys, combination))
            
            # Run the function
            try:
                # Assuming the function signature is (npy_array, settings, **kwargs)
                candidates, metadata = func(filtered_events, settings, **kwargs)
                
                # Evaluate
                # Using number of candidates as a placeholder metric
                score = len(candidates) if candidates is not None else 0
                
                if score > best_score:
                    best_score = score
                    best_params = kwargs
                    best_method = method_name
                    print(f"  New best: {score} candidates with params {kwargs}")
                    
            except Exception as e:
                logging.error(f"Error running {method_name} with {kwargs}: {e}")

    print("\nSearch Complete.")
    print(f"Best Method: {best_method}")
    print(f"Best Params: {best_params}")
    print(f"Best Score: {best_score}")
    
    return best_method, best_params

if __name__ == "__main__":
    # Placeholder for running directly if needed
    pass
