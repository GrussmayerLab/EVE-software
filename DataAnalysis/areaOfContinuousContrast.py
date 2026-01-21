import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter, sobel
import os
from concurrent.futures import ProcessPoolExecutor

def __function_metadata__():
    return {
        'run_analysis': {
            'display_name': 'Area of Continuous Contrast',
            'help_string': 'Calculate the area of continuous contrast for a given event dataset',
            'required_kwargs': [
                {'name': 'x_res', 'type': int, 'default': 941, 'description': 'Sensor resolution (X-Axis) in microns.'},
                {'name': 'y_res', 'type': int, 'default': 483, 'description': 'Sensor resolution (Y-Axis) in microns.'}
            ],
            'optional_kwargs': [
                {'name': 'min_interval', 'type': int, 'default': None, 'description': 'Min interval (default: auto)'},
                {'name': 'max_interval', 'type': int, 'default': None, 'description': 'Max interval (default: auto)'},
                {'name': 'step_interval', 'type': int, 'default': None, 'description': 'Step size (default: auto)'},
            ],
        }
    }

def compute_single_interval(args):
    """
    Worker function to process a single interval.
    Args are packed in a tuple to be compatible with executor.map
    """
    interval, t, x, y, width, height = args
    
    # 1. Define Time Bins
    t_start, t_end = t[0], t[-1]
    bins = np.arange(t_start, t_end + interval, interval)
    
    # 2. Assign events to frames 
    frame_indices = np.digitize(t, bins) - 1
    
    # Filter valid frames
    valid_mask = (frame_indices >= 0) & (frame_indices < (len(bins) - 1))
    
    if not np.any(valid_mask):
        return {'interval': interval, 'mean_contrast': 0.0}

    unique_frames = np.unique(frame_indices[valid_mask])
    contrasts = []
    
    # 3. Process each frame
    for f_idx in unique_frames:
        mask = frame_indices == f_idx
        
        # Fast histogram
        img, _, _ = np.histogram2d(y[mask], x[mask], 
                                   bins=[height, width], 
                                   range=[[0, height], [0, width]])
        
        img = (img > 0).astype(np.float32) * 255

        blurred = gaussian_filter(img, sigma=2)
        
        grad_x = sobel(blurred, axis=1)
        grad_y = sobel(blurred, axis=0)
        magnitude = np.hypot(grad_x, grad_y)
        contrasts.append(np.std(magnitude))

    mean_val = np.mean(contrasts) if contrasts else 0.0
    return {'interval': interval, 'mean_contrast': mean_val}


def run_analysis(ev, x_res=256, y_res=256, min_interval=None, max_interval=None, step_interval=None):
    """
    Run the area of continuous contrast analysis on the given event dataset, done in parallel for multiple intervals.
    """
    x_res = int(x_res)
    y_res = int(y_res)
    
    t = ev['t'].astype(np.float64)
    x = ev['x'].astype(np.int32)
    y = ev['y'].astype(np.int32)
    
    max_x_event = x.max()
    max_y_event = y.max()
    
    if max_x_event >= x_res:
        print(f"DEBUG: Auto-adjusting X resolution from {x_res} to {max_x_event + 1}")
        x_res = int(max_x_event + 1)
        
    if max_y_event >= y_res:
        print(f"DEBUG: Auto-adjusting Y resolution from {y_res} to {max_y_event + 1}")
        y_res = int(max_y_event + 1)

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
        step_interval = int((max_interval - min_interval) / 50)
        step_interval = max(100, step_interval)
    else:
        step_interval = int(step_interval)
    
    if duration < min_interval:
        print("WARNING: Interval is larger than total recording duration!")
        return None, {'area': 0}

    print(f"Auto-Configured Analysis: Min={min_interval}, Max={max_interval}, Step={step_interval}")

    intervals = np.arange(min_interval, max_interval, step_interval)
    
    tasks = [(iv, t, x, y, x_res, y_res) for iv in intervals]
    
    results_list = []
    print(f"Starting parallel processing with {os.cpu_count()} workers...")
    
    with ProcessPoolExecutor() as executor:
        results_gen = executor.map(compute_single_interval, tasks)
        results_list = list(results_gen)

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
    return plt.gcf(), {'area': area}