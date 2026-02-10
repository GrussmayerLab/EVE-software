import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter, sobel
import os
import gc
from concurrent.futures import ProcessPoolExecutor
import random

# --- Global container for worker processes ---
# This ensures data is not pickled/copied for every single task
worker_data = {}

def init_worker(t_shared, x_shared, y_shared):
    """
    Initialize the worker process by storing the large arrays in global memory.
    This runs once per process, not once per task.
    """
    worker_data['t'] = t_shared
    worker_data['x'] = x_shared
    worker_data['y'] = y_shared

def __function_metadata__():
    return {
        'run_analysis': {
            'display_name': 'Area of Continuous Contrast',
            'help_string': 'Calculate the area of continuous contrast for a given event dataset',
            'required_kwargs': [
                {'name': 'x_res', 'type': int, 'default': 1043, 'description': 'Sensor resolution (X-Axis) in microns.'},
                {'name': 'y_res', 'type': int, 'default': 405, 'description': 'Sensor resolution (Y-Axis) in microns.'}
            ],
            'optional_kwargs': [
                {'name': 'min_interval', 'type': int, 'default': 10000, 'description': 'Min interval (default: auto)'},
                {'name': 'max_interval', 'type': int, 'default': 50000000, 'description': 'Max interval (default: auto)'},
                {'name': 'step_interval', 'type': int, 'default': None, 'description': 'Step size (default: auto)'},
            ],
        }
    }

def compute_interval_batch(args):
    """
    Worker function to process a batch of intervals.
    Args: (intervals_chunk, width, height)
    """
    intervals_chunk, width, height = args
    
    # Retrieve data from global storage (zero copy)
    t = worker_data['t']
    x = worker_data['x']
    y = worker_data['y']
    
    t_start, t_end = t[0], t[-1]
    
    results = []
    
    for interval in intervals_chunk:
        # 1. Define Time Bins
        # RAM OPTIMIZATION: Instead of np.digitize, use searchsorted
        bins = np.arange(t_start, t_end + interval, interval)
        
        # Find indices where time bins start/end
        idx_bounds = np.searchsorted(t, bins)
        
        contrasts = []
        
        # 2. Process each frame using slices
        for i in range(len(idx_bounds) - 1):
            start_idx = idx_bounds[i]
            end_idx = idx_bounds[i+1]
            
            # Skip empty frames
            if start_idx == end_idx:
                continue
                
            # Create views (zero copy) of the current frame's events
            x_slice = x[start_idx:end_idx]
            y_slice = y[start_idx:end_idx]
            
            # Fast histogram using bincount (approx 10x faster than histogram2d)
            # Filter out-of-bounds events first (histogram2d does this implicitly via range)
            mask = (x_slice >= 0) & (x_slice < width) & (y_slice >= 0) & (y_slice < height)
            if not mask.all():
                x_s = x_slice[mask]
                y_s = y_slice[mask]
            else:
                x_s = x_slice
                y_s = y_slice

            if len(x_s) > 0:
                flat_indices = y_s * width + x_s
                # bincount is much faster
                img_flat = np.bincount(flat_indices, minlength=width*height)
                img = img_flat.reshape((height, width))
                
                # Convert to boolean contrast map
                img = (img > 0).astype(np.float32) * 255

                # 3. Contrast Calculation
                blurred = gaussian_filter(img, sigma=2)
                grad_x = sobel(blurred, axis=1)
                grad_y = sobel(blurred, axis=0)
                magnitude = np.hypot(grad_x, grad_y)
                contrasts.append(np.std(magnitude))
                
                # cleanup per frame
                del img, blurred, grad_x, grad_y, magnitude
            else:
                # No events in this frame (or all were out of bounds)
                contrasts.append(0.0)

        mean_val = np.mean(contrasts) if contrasts else 0.0
        results.append({'interval': interval, 'mean_contrast': mean_val})
    
    return results


def run_analysis(ev, x_res=256, y_res=256, min_interval=None, max_interval=None, step_interval=None):
    """
    Run the area of continuous contrast analysis optimized for memory usage.
    """
    x_res = int(x_res)
    y_res = int(y_res)
    
    # Ensure contiguous arrays for better performance
    t = np.ascontiguousarray(ev['t'].astype(np.float64))
    x = np.ascontiguousarray(ev['x'].astype(np.int32))
    y = np.ascontiguousarray(ev['y'].astype(np.int32))
    
    # max_x_event = x.max()
    # max_y_event = y.max()
    
    # if max_x_event >= x_res:
    #     print(f"DEBUG: Auto-adjusting X resolution from {x_res} to {max_x_event + 1}")
    #     x_res = int(max_x_event + 1)
        
    # if max_y_event >= y_res:
    #     print(f"DEBUG: Auto-adjusting Y resolution from {y_res} to {max_y_event + 1}")
    #     y_res = int(max_y_event + 1)

    duration = t[-1] - t[0]
    print(f"Recording Duration: {duration} | Resolution: {x_res}x{y_res}")

    if min_interval is None:
        min_interval = max(1000, int(duration * 0.001))
    else:
        min_interval = int(min_interval)
        
    if max_interval is None:
        max_interval = int(duration * 0.2)
    else:
        max_interval = int(max_interval)
        
    if step_interval is None:
        step_interval = int((max_interval - min_interval) / 100)
        step_interval = max(100, step_interval)
    else:
        step_interval = int(step_interval)
    
    if duration < min_interval:
        print("WARNING: Interval is larger than total recording duration!")
        return None, {'area': 0}

    print(f"Auto-Configured Analysis: Min={min_interval}, Max={max_interval}, Step={step_interval}")

    intervals = np.arange(min_interval, max_interval, step_interval)
    
    # MEMORY FIX: Only pass metadata in the tasks list
    # t, x, y are NOT passed here
    
    # Create batches of tasks to reduce IPC overhead
    num_workers = os.cpu_count()
    total_intervals = len(intervals)
    # Heuristic: 50 batches per worker to balance load allowing work stealing if needed,
    # but kept reasonable to avoid IPC overhead
    min_batch_size = 10
    batch_size = max(min_batch_size, int(total_intervals / (num_workers * 10))) 
    
    tasks = []
    # Shuffle intervals to balance load across workers
    # Small intervals = many frames = slow. Large intervals = few frames = fast.
    # Without shuffling, the first worker gets all the heavy queries and runs forever.
    intervals_shuffled = intervals.copy()
    np.random.shuffle(intervals_shuffled)
    
    for i in range(0, total_intervals, batch_size):
        chunk = intervals_shuffled[i : i + batch_size]
        tasks.append((chunk, x_res, y_res))
    
    results_list = []
    
    print(f"Starting parallel processing with {num_workers} workers (Batch size: {batch_size}, Total batches: {len(tasks)})...")
    
    with ProcessPoolExecutor(max_workers=num_workers, 
                             initializer=init_worker, 
                             initargs=(t, x, y)) as executor:
        results_gen = executor.map(compute_interval_batch, tasks)
        
        # Flatten the list of lists
        for batch_result in results_gen:
            results_list.extend(batch_result)

    results = pd.DataFrame(results_list)
    results = results.sort_values('interval')
    
    if results.empty or results['mean_contrast'].sum() == 0:
        print("WARNING: All contrast values are 0.0. Check your x/y coordinates or kernel size.")

    plt.figure(figsize=(10, 6))
    plt.plot(results['interval'], results['mean_contrast'], marker='o', markersize=2)
    plt.title(f'Contrast Analysis ({x_res}x{y_res})')
    plt.xlabel('Interval (us)')
    plt.ylabel('Contrast')
    plt.grid(True)
    
    area = trapezoid(results['mean_contrast'], results['interval'])
    return plt.gcf(), {
        'area': area,
        'curve_x': results['interval'].tolist(),
        'curve_y': results['mean_contrast'].tolist()
    }