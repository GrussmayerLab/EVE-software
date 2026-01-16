import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter, sobel

def __function_metadata__():
    return {
        'run_analysis': {
            'display_name': 'Area of Continuous Contrast',
            'help_string': 'Calculate the area of continuous contrast for a given event dataset',
            'required_kwargs': [
                {'name': 'x_res', 'type': int, 'default': 256, 'description': 'Sensor resolution (X-Axis) in microns.'},
                {'name': 'y_res', 'type': int, 'default': 256, 'description': 'Sensor resolution (Y-Axis) in microns.'},
                {'name': 'min_interval', 'type': int, 'default': 20000, 'description': 'Minimum interval for contrast calculation'},
                {'name': 'max_interval', 'type': int, 'default': 30000, 'description': 'Maximum interval for contrast calculation'},
                {'name': 'step_interval', 'type': int, 'default': 1000, 'description': 'Step interval for contrast calculation'},
            ],
            'optional_kwargs': [],
        }
    }

def run_analysis(ev, x_res, y_res, min_interval=2000, max_interval=200001, step_interval=2000):
    min_interval = int(min_interval)
    max_interval = int(max_interval)
    step_interval = int(step_interval)
    min_value = 0
    max_value = max_interval - 1
    x_res = int(x_res)
    y_res = int(y_res)

    # find number of events
    count=30000

    # Auto-adjust resolution if events are out of bounds
    if len(ev) > 0:
        x_res = max(x_res, int(ev['x'].max()) + 1)
        y_res = max(y_res, int(ev['y'].max()) + 1)
    print(ev.shape)
    if len(ev) < 2 * count: return 0.5
    contrasts = []
    rows = []
    for interval in np.arange(min_interval, max_interval, step_interval):
        accumulation_images = create_accumulation_images(ev, x_res, y_res, interval)
        contrasts = [calculate_contrast(apply_gaussian_blur(frame,2,5)) for frame in accumulation_images]
        mean_contrast, median_contrast, rms_contrast = compute_statistics(contrasts)
        rows.append({
            'interval': interval,
            'mean_contrast': mean_contrast,
            'median_contrast': median_contrast,
            'rms_contrast': rms_contrast
        })
    results = pd.DataFrame(rows)
    print(results)
    plot, area = plot_contrast_statistics(results, min_value, max_value)

    return plot, {'area': area}


def create_accumulation_images(events, width, height, interval):
    """
    Create time-based accumulation frames from event data.

    Args:
        events (list): List of event tuples (t, x, y, p)
        width (int): Width of the output image
        height (int): Height of the output image
        interval (float): Time interval for accumulating events

    Returns:
        list: List of accumulated event frames
    """
    frames = []
    current_frame = np.zeros((height, width), dtype=np.uint8)
    last_time = None
    frame_count = 0

    for t, x, y, p in events:
        if last_time is None:
            last_time = t

        while t - last_time >= interval:
            if np.any(current_frame):
                frames.append(current_frame.copy())
                frame_count += 1
            current_frame = np.zeros((height, width), dtype=np.uint8)
            last_time += interval

        current_frame[y, x] = 255

    if np.any(current_frame):
        frames.append(current_frame)

    return frames


def apply_gaussian_blur(image, kernel_size=5, sigma=2):
    """
    Apply Gaussian blur to an image.

    Args:
        image (numpy.ndarray): Input image
        kernel_size (int): Size of the Gaussian kernel
        sigma (float): Standard deviation of the Gaussian kernel

    Returns:
        numpy.ndarray: Blurred image
    """
    return gaussian_filter(image.astype(float), sigma=sigma)


def calculate_contrast(image):
    """
    Calculate image contrast using SciPy Sobel operators.
    """
    # If image is 3D (RGB), convert to grayscale (Luminance formula)
    if len(image.shape) == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Calculate gradients
    grad_x = sobel(image, axis=1)
    grad_y = sobel(image, axis=0)

    # Gradient magnitude
    magnitude = np.hypot(grad_x, grad_y)

    # Contrast is the standard deviation of the magnitude
    return np.std(magnitude)


def compute_statistics(contrasts):
    """
    Compute statistical measures from contrast values.

    Args:
        contrasts (list): List of contrast values

    Returns:
        tuple: (mean, median, root mean square) of contrast values
    """
    mean_val = np.mean(contrasts)
    median_val = np.median(contrasts)
    rms_val = np.sqrt(np.mean(np.square(contrasts)))
    return mean_val, median_val, rms_val


def calculate_area_under_curve(results, y_column, x_column, x_min, x_max):
    """
    Calculate the area under the curve within specified x-axis bounds.

    Args:
        results (pd.DataFrame): DataFrame containing the results
        y_column (str): Name of the y-axis column
        x_column (str): Name of the x-axis column
        x_min (float): Minimum x value for integration
        x_max (float): Maximum x value for integration

    Returns:
        float: Area under the curve
    """

    # Filter data based on x-axis range
    mask = (results[x_column] >= x_min) & (results[x_column] <= x_max)
    filtered_df = results.loc[mask].sort_values(by=x_column)

    if filtered_df.empty:
        return 0.0
    # Calculate area using trapezoidal rule
    area = trapezoid(filtered_df[y_column], filtered_df[x_column])
    return area


def plot_contrast_statistics(results, min_value, max_value):
    """
    Plot contrast statistics for multiple CSV files and calculate areas under curves.

    Args:
        results (pd.DataFrame): DataFrame containing the results
        min_value (float): Minimum x value for area calculation
        max_value (float): Maximum x value for area calculation
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results['interval'], results['mean_contrast'], label='mean_contrast',
              marker='o', markersize=1)
    area = calculate_area_under_curve(results,'mean_contrast', 'interval',
                                      min_value, max_value)



    plt.title('Contrast Statistics Over Different Intervals')
    plt.xlabel('interval')
    plt.ylabel('contrast')
    plt.legend()
    plt.grid(True)
    return plt.gcf(), area