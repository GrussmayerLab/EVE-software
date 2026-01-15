
import matplotlib.pyplot as plt
import numpy as np

def __function_metadata__():
    return {
        'run_analysis': {
            'display_name': 'Example Analysis',
            'help_string': 'A dummy analysis function that generates a random plot and some metrics.',
            'required_kwargs': [
                {'name': 'param1', 'type': float, 'default': 1.0, 'description': 'A multiplier parameter.'},
                {'name': 'title', 'type': str, 'default': 'My Plot', 'description': 'Title of the plot.'}
            ],
            'optional_kwargs': [],
            # Removed empty dictionary kwargs which can cause issues with legacy utils functions
            #'dist_kwarg': {},
            #'time_kwarg': {}
        }
    }

def run_analysis(data=None, settings=None, param1=1.0, title='My Plot'):
    """
    Dummy analysis function.
    """
    # Cast inputs to correct types as they might be passed as strings from GUI
    try:
        param1 = float(param1)
    except:
        param1 = 1.0
        
    # Create a figure
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * param1
    ax.plot(x, y)
    ax.set_title(str(title))
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

    # Create dummy results
    results = {
        'max_amplitude': float(np.max(np.abs(y))),
        'parameter_used': float(param1),
        'data_entries': len(data) if data is not None else 0
    }

    return fig, results
