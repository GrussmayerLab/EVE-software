
import matplotlib.pyplot as plt
import numpy as np

def __function_metadata__():
    return {
        'run_analysis': {
            'display_name': '2D Projection',
            'help_string': 'Generate a 2D projection plot of the data events.',
            'required_kwargs': [
                {'name': 'Positive Events', 'type': bool, 'default': True, 'description': 'Include positive polarity events.'},
                {'name': 'Negative Events', 'type': bool, 'default': True, 'description': 'Include negative polarity events.'},
                # {'name': 'Coloring', 'type': str, 'default': 'By Polarity', 'description': 'Coloring scheme for the plot (e.g., By Polarity, By Time).'},
            ],
            'optional_kwargs': [],
            # Removed empty dictionary kwargs which can cause issues with legacy utils functions
            #'dist_kwarg': {},
            #'time_kwarg': {}
        }
    }


def run_analysis(ev, positive_events=True, negative_events=True, plot_type='accumulation'):
    if not positive_events and not negative_events:
        raise ValueError("At least one of Positive Events or Negative Events must be True.")

    # Filter events based on polarity
    # ev format: [ts, x, y, p]
    if positive_events and not negative_events:
        ev = ev[ev[:, 3] > 0]
    elif negative_events and not positive_events:
        ev = ev[ev[:, 3] < 0]

    # Determine sensor resolution
    width = int(np.max(ev[:, 1])) + 1
    height = int(np.max(ev[:, 2])) + 1

    plt.figure(figsize=(10, 8))

    if plot_type == 'accumulation':
        # Create a 2D histogram to count events per pixel
        # bins correspond to pixel coordinates
        img, x_edges, y_edges = np.histogram2d(
            ev[:, 1], ev[:, 2],
            bins=[width, height],
            range=[[0, width], [0, height]]
        )

        # We transpose the image because histogram2d uses (x, y)
        # but imshow expects (row, col) which is (y, x)
        plt.imshow(img.T, origin='lower', cmap='hot', interpolation='nearest')
        plt.colorbar(label='Event Count (Density)')
        plt.title('2D Accumulation (Density Map)')

    elif plot_type == 'time_surface':
        # Time surface usually shows the LATEST timestamp at each pixel
        img = np.zeros((height, width))
        # Fill the array with timestamps; later events overwrite earlier ones
        for e in ev:
            img[int(e[2]), int(e[1])] = e[0]

        plt.imshow(img, origin='lower', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Timestamp (Recency)')
        plt.title('Time Surface (Last Event)')

    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()


if __name__ == '__main__':
    # Simulated data: [timestamp, x, y, polarity]
    num_events = 50000
    events = np.random.rand(num_events, 4)
    events[:, 0] *= 10.0  # 10 seconds
    events[:, 1] *= 640  # Width
    events[:, 2] *= 480  # Height
    events[:, 3] = np.where(events[:, 3] > 0.5, 1, -1)

    # Run fixed analysis
    run_analysis(events, plot_type='time_surface')