import inspect
try:
    from eve_smlm.Utils import utilsHelper
except ImportError:
    from Utils import utilsHelper
import pandas as pd
import numpy as np
import time, logging
from scipy import spatial
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from joblib import Parallel, delayed
from scipy.spatial import ConvexHull
import open3d as o3d
from scipy.optimize import curve_fit, root
import numexpr as ne

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Required function __function_metadata__
def __function_metadata__():
    return {
        "spatio_temporal_recursive_filtering": {
            "required_kwargs": [
                {"name": "tau", "display_text": "Tau (s)", "description": "Time constant for decay (in seconds).", "default": 0.05, "type": float},
                {"name": "filter_size", "display_text": "Filter Size (px)", "description": "Size of the spatial Gaussian kernel (must be odd).", "default": 3, "type": int},
                {"name": "sampling_threshold", "display_text": "Sampling Threshold", "description": "Threshold factor for probabilistic subsampling (0.0 to 1.0+).", "default": 1.0, "type": float},
                {"name": "normalization_length", "display_text": "Normalization Length", "description": "Window length for local normalization of filter values.", "default": 1000, "type": int},
            ],
            "optional_kwargs": [
                {"name": "image_width", "display_text": "Image Width", "description": "Sensor width. 0 to auto-detect.", "default": 0, "type": int},
                {"name": "image_height", "display_text": "Image Height", "description": "Sensor height. 0 to auto-detect.", "default": 0, "type": int},
                {"name": "time_unit_multiplier", "display_text": "Time Unit Multiplier", "description": "Multiplier to convert Tau (s) to data time units (e.g., 1000 for ms, 1e6 for us).", "default": 1000.0, "type": float},
                {"name": "debug", "display_text": "Debug Boolean", "description": "Get some debug info.", "default": False},
            ],
            "help_string": "Causal DB Subsampling. Applies a recursive filter that accumulates events over time with exponential decay (tau) and convolves them with a spatial Gaussian kernel. High-response events are retained based on a sampling threshold.",
            "display_name": "Causal DB Subsampling"
        }
    }

#-------------------------------------------------------------------------------------------------------------------------------
# Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

def generate_gaussian_kernel(filter_size):
    """
    Generates a normalized 2D Gaussian kernel.
    """
    if filter_size % 2 == 0:
        logging.warning(f"Filter size {filter_size} is even. Increasing by 1.")
        filter_size += 1
        
    sigma = filter_size / 5.0
    kernel = np.zeros((filter_size, filter_size))
    kernel[filter_size // 2, filter_size // 2] = 1
    gaussian_kernel = gaussian_filter(kernel, sigma)
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    return gaussian_kernel

def normalize_filter_values(filter_values, norm_length):
    """
    Normalizes filter values using a sliding window approach.
    """
    filter_values_normalized = np.zeros_like(filter_values)
    
    
    for i in range(len(filter_values)):
        start = max(0, i - norm_length)
        chunk = filter_values[start : i+1]
        
        c_min = np.min(chunk)
        c_max = np.max(chunk)
        
        val = (chunk[-1] - c_min) / (c_max - c_min + 1e-6)
        filter_values_normalized[i] = val
        
    return filter_values_normalized

def calculate_recursive_filter(x_arr, y_arr, t_arr, p_arr, H, W, tau, kernel, filter_size):
    """
    Core recursive filtering logic.
    """
    K = filter_size // 2
    num_events = len(t_arr)
    
    # Tensors
    last_time_tensor = np.full((2, H, W), float('-inf'), dtype=np.float32) # Init to -inf for correct first diff
    # If using 0 as init (per original code), the first exp decay might be weird if t is large, 
    # but we stick to original logic:
    last_time_tensor = np.full((2, H, W), 0.0, dtype=np.float32)
    
    temporal_accumulation_tensor = np.zeros((2, H, W), dtype=np.float32)
    filter_values = np.zeros(num_events, dtype=np.float32)

    # Pre-compute kernel dimensions
    k_h, k_w = kernel.shape

    # Main Loop
    # Note: This loop is computationally expensive in pure Python.
    for i in range(num_events):
        x = int(x_arr[i])
        y = int(y_arr[i])
        t = float(t_arr[i])
        p = int(p_arr[i])
        
        # Polarity index (ensure 0 or 1)
        pp = 0 if p < 0 else (1 if p > 0 else 0) 
        # Or if input is 0/1:
        pp = p if p in [0, 1] else 0

        # Boundary checks
        h_start = max(y - K, 0)
        h_end = min(y + K, H - 1)
        w_start = max(x - K, 0)
        w_end = min(x + K, W - 1)

        # Slice for the kernel (handle image boundaries)
        k_h_start = K - (y - h_start)
        k_h_end = K + (h_end - y) + 1
        k_w_start = K - (x - w_start)
        k_w_end = K + (w_end - x) + 1
        
        # Current patch in accumulation tensor
        acc_view = temporal_accumulation_tensor[pp, h_start:h_end+1, w_start:w_end+1]
        last_t_view = last_time_tensor[pp, h_start:h_end+1, w_start:w_end+1]

        # Compute temporal lag
        # exp( - (t - last_t) / tau )
        dt = t - last_t_view
        temporal_lag = np.exp(-dt / tau)

        # Update last time
        last_time_tensor[pp, h_start:h_end+1, w_start:w_end+1] = t

        # Update accumulation
        # In-place update: acc *= lag
        acc_view[:] *= temporal_lag
        
        # Add 1 to the specific pixel
        temporal_accumulation_tensor[pp, y, x] += 1

        # Compute filter value: Sum(Accumulation * Kernel)
        # We need to slice the kernel to match the boundary-clipped image patch
        kernel_slice = kernel[k_h_start:k_h_end, k_w_start:k_w_end]
        
        # Convolve
        val = np.sum(acc_view * kernel_slice)
        filter_values[i] = val

    return filter_values

#-------------------------------------------------------------------------------------------------------------------------------
# Main Analysis Function
#-------------------------------------------------------------------------------------------------------------------------------

def spatio_temporal_recursive_filtering(npy_array, settings, **kwargs):
    """
    Spatio-temporal recursive filtering and subsampling.
    """
    # Check arguments
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(
        __function_metadata__(), inspect.currentframe().f_code.co_name, kwargs
    ) #type:ignore

    starttime = time.time()
    logging.info(f"Starting Spatio-Temporal Recursive Filtering. {len(npy_array)} events.")
    logging.info(f"Parameters: {kwargs}")
    
    # 1. Parse Parameters
    tau_seconds = float(kwargs['tau'])
    time_mult = float(kwargs.get('time_unit_multiplier', 1000.0))
    tau = tau_seconds * time_mult # Convert tau to match data time units
    
    filter_size = int(kwargs['filter_size'])
    sampling_threshold = float(kwargs['sampling_threshold'])
    norm_length = int(kwargs['normalization_length'])
    
    # 2. Prepare Data
    # Sort by time just in case, though usually pre-sorted
    if not np.all(npy_array['t'][:-1] <= npy_array['t'][1:]):
        npy_array = np.sort(npy_array, order='t')

    x_data = npy_array['x'].astype(int)
    y_data = npy_array['y'].astype(int)
    t_data = npy_array['t'].astype(float)
    p_data = npy_array['p'].astype(int)

    # Determine Image Size
    W_in = int(kwargs.get('image_width', 0))
    H_in = int(kwargs.get('image_height', 0))
    
    if W_in <= 0:
        W = int(np.max(x_data)) + 1
    else:
        W = W_in
        
    if H_in <= 0:
        H = int(np.max(y_data)) + 1
    else:
        H = H_in

    logging.info(f"Recursive Filter: Tau={tau}, Grid=({H}x{W}), Events={len(t_data)}")

    # 3. Kernel Generation
    gaussian_kernel = generate_gaussian_kernel(filter_size)

    # 4. Run Recursive Filter
    # Note: Running this on Python for >100k events will be slow. 
    # Numba is recommended here but not strictly available in standard imports provided.
    filter_values = calculate_recursive_filter(
        x_data, y_data, t_data, p_data, 
        H, W, tau, gaussian_kernel, filter_size
    )

    # 5. Normalize
    if norm_length > 0:
        filter_values = normalize_filter_values(filter_values, norm_length)

    # 6. Subsampling
    # Generate random probabilities
    rng = np.random.default_rng()
    probs = rng.random(len(filter_values))
    
    # Indices where random < threshold * value
    # If value is high (salient event), threshold * value is high, more likely to be kept.
    mask = probs < (sampling_threshold * filter_values)
    
    filtered_events = npy_array[mask]
    
    # 7. Format Output
    # Convert structured array to DataFrame for the 'candidates' format
    df = pd.DataFrame(filtered_events)
    
    candidates = {}
    # We treat the filtered result as a single "cluster" or "candidate" set (ID 0)
    # This allows it to be passed to subsequent clustering steps or visualizers.
    if len(df) > 0:
        candidates[0] = {}
        candidates[0]['events'] = df
        candidates[0]['N_events'] = len(df)
        candidates[0]['cluster_size'] = [
            int(df['y'].max() - df['y'].min() + 1),
            int(df['x'].max() - df['x'].min() + 1),
            int(df['t'].max() - df['t'].min())
        ]
    else:
        logging.warning("Spatio-temporal filtering removed all events.")

    duration = time.time() - starttime
    performance_metadata = f"Spatio-Temporal Recursive Filter ran for {duration:.4f} seconds. Kept {len(df)}/{len(npy_array)} events."
    
    return candidates, performance_metadata