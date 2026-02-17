import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def __function_metadata__():
    return {
        'run_analysis': {
            'display_name': 'Dataset Statistics',
            'help_string': 'Display basic statistics about the event dataset including counts, rates, and density.',
            'required_kwargs': [
                {'name': 'x_res', 'type': int, 'default': 0, 'description': 'Sensor Width'},
                {'name': 'y_res', 'type': int, 'default': 0, 'description': 'Sensor Height'}
            ],
            'optional_kwargs': [],
        }
    }

def run_analysis(ev, x_res=0, y_res=0):
    """
    Analyzes the event packet and prints/returns statistics.
    """
    x_res = int(x_res)
    y_res = int(y_res)
    
    if len(ev) > 0:
        max_x = int(ev['x'].max())
        max_y = int(ev['y'].max())
        if max_x >= x_res:
            print(f"DEBUG: Auto-adjusting X resolution from {x_res} to {max_x + 1}")
            x_res = max_x + 1
        if max_y >= y_res:
            print(f"DEBUG: Auto-adjusting Y resolution from {y_res} to {max_y + 1}")
            y_res = max_y + 1

    stats = {}
    
    # 1. Basic Counts
    total_events = len(ev)
    stats['total_events'] = total_events
    
    if total_events == 0:
        print("Dataset is empty.")
        return plt.figure(), stats

    # 2. Time Statistics
    t_start = ev['t'][0]
    t_end = ev['t'][-1]
    duration_us = t_end - t_start
    duration_s = duration_us / 1e6
    stats['duration_sec'] = duration_s
    
    # Avoiding division by zero
    if duration_s > 0:
        ev_rate_keps = (total_events / duration_s) / 1000
    else:
        ev_rate_keps = 0
    stats['rate_keps'] = ev_rate_keps

    # 3. Spatial Statistics
    # Calculate unique active pixels
    # Map 2D coordinates to 1D index
    linear_indices = ev['x'].astype(np.int64) + ev['y'].astype(np.int64) * x_res
    unique_pixels = np.unique(linear_indices).size
    total_pixels = x_res * y_res
    fill_factor = (unique_pixels / total_pixels) * 100
    stats['active_pixels'] = unique_pixels
    stats['fill_factor_percent'] = fill_factor

    # Density / Area metrics
    pixel_counts = np.bincount(linear_indices, minlength=total_pixels)
    
    max_density = pixel_counts.max()
    mean_density = pixel_counts.mean()
    std_density = pixel_counts.std()
    
    stats['max_density'] = max_density
    stats['mean_density'] = mean_density
    stats['std_density'] = std_density
    

    # 4. Polarity Statistics
    if 'p' in ev.dtype.names:
        p_on = np.count_nonzero(ev['p'] == 1)
        p_off = np.count_nonzero(ev['p'] == 0)
        on_off_ratio = p_on / p_off if p_off > 0 else 0
        stats['p_on'] = p_on
        stats['p_off'] = p_off
        stats['on_off_ratio'] = on_off_ratio
    else:
        p_on = 0
        p_off = 0
        stats['p_on'] = 0
        on_off_ratio = 0
        
    # 5. Refractory Period Estimation
    try:
        df = pd.DataFrame(ev)

        df = df.sort_values(['x', 'y', 't'])
        
        pixel_changed = (df['x'] != df['x'].shift()) | (df['y'] != df['y'].shift())
        
        df['dt'] = df['t'].diff()
        
        df.loc[pixel_changed, 'dt'] = np.nan
        
        valid_dts = df[df['dt'] > 0]
        
        if not valid_dts.empty:
            # Find min dt for each pixel
            min_dts_per_pixel = valid_dts.groupby(['x', 'y'])['dt'].min()
            
            stats['refr_min_us'] = min_dts_per_pixel.min()
            stats['refr_mean_us'] = min_dts_per_pixel.mean()
            stats['refr_median_us'] = min_dts_per_pixel.median()
            stats['refr_std_us'] = min_dts_per_pixel.std()
        else:
             min_dts_per_pixel = pd.Series(dtype=float)
             stats['refr_mean_us'] = 0
        
    except Exception as e:
        print(f"Error calculating refractory period: {e}")
        min_dts_per_pixel = pd.Series(dtype=float)
        

    # --- Visualization ---
    # Create a dashboard-like figure
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3)
    
    # Plot 1: Pixel Density Distribution (Histogram of pixel counts)
    ax1 = fig.add_subplot(gs[0, 0])
    # Filter out zero-count pixels to see the distribution of usage better.
    active_pixel_counts = pixel_counts[pixel_counts > 0]
    if len(active_pixel_counts) > 0:
        ax1.hist(active_pixel_counts, bins=50, log=True, color='tab:purple', edgecolor='black', alpha=0.7)
        ax1.set_title(f'Event Density Distribution\n(Active Pixels Only)')
        ax1.set_xlabel('Events per Pixel')
        ax1.set_ylabel('Count of Pixels (Log)')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No Events", ha='center')

    # Plot 2: Polarity Balance
    ax2 = fig.add_subplot(gs[0, 1])
    if 'p' in ev.dtype.names and (p_on + p_off > 0):
        ax2.pie([p_on, p_off], labels=['ON', 'OFF'], colors=['#4CAF50', '#F44336'], autopct='%1.1f%%', startangle=90)
        ax2.set_title('Polarity Balance')
    else:
        ax2.text(0.5, 0.5, "No Polarity Data", ha='center')
        ax2.axis('off')

    # Plot 3: Refractory Period Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if not min_dts_per_pixel.empty:
        ax3.hist(min_dts_per_pixel, bins=50, color='tab:orange', edgecolor='black', alpha=0.7)
        ax3.set_title(f'Est. Refractory Period\n(Min dt per Pixel)')
        ax3.set_xlabel('Time (us)')
        ax3.set_ylabel('Count of Pixels')
        ax3.grid(True, alpha=0.3)
        
        stats_text = (f"Min: {stats.get('refr_min_us', 0):.1f} us\n"
                      f"Mean: {stats.get('refr_mean_us', 0):.1f} us\n"
                      f"Median: {stats.get('refr_median_us', 0):.1f} us")
        ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes, 
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    
    else:
        ax3.text(0.5, 0.5, "Not enough data for Refr. Period", ha='center')

    plt.tight_layout()
    
    return fig, stats

if __name__ == "__main__":
    # Test block
    dtype = [('x', 'u2'), ('y', 'u2'), ('t', 'u8'), ('p', 'u1')]
    N = 100000
    ev = np.zeros(N, dtype=dtype)
    ev['x'] = np.random.randint(0, 640, N)
    ev['y'] = np.random.randint(0, 480, N)
    ev['t'] = np.sort(np.random.randint(0, 10000000, N)).astype('u8') # 10s
    ev['p'] = np.random.randint(0, 2, N)
    
    run_analysis(ev, 640, 480)
    plt.show()
