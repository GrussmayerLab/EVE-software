import numpy as np
import matplotlib.pyplot as plt


def __function_metadata__():
    return {
        'run_analysis': {
            'display_name': 'Event Structural Ratio',
            'help_string': 'Calculate the event structural ratio for a given event dataset',
            'required_kwargs': [
                {'name': 'X-res', 'type': int, 'default': 256, 'description': 'Sensor resolution (X-Axis) in microns.'},
                {'name': 'Y-res', 'type': int, 'default': 256, 'description': 'Sensor resolution (Y-Axis) in microns.'},
                {'name': 'M', 'type': int, 'default': 20000, 'description': 'Reference number of events for interpolation'},
                {'name': 'N', 'type': int, 'default': 30000, 'description': 'Number of events in the dataset'}
            ],
            'optional_kwargs': [],
            # Removed empty dictionary kwargs which can cause issues with legacy utils functions
            #'dist_kwarg': {},
            #'time_kwarg': {}
        }
    }

def run_analysis(ev, x_res, y_res, count=30000, refN=20000):
    if len(ev) < 2 * count: return 0.5
    score = np.zeros(int(len(ev)/count) - 1)

    for i in range(0, len(score)):
        st_idx = i * count
        ed_idx = st_idx + count
        packet = ev[st_idx:ed_idx]

        cnt = count_distribution(packet, [x_res, y_res], use_polarity=False)

        N = cnt.sum()
        L = cnt.size - ((1 - refN/N) ** cnt).sum()

        score[i] = (cnt * (cnt-1) / (N + np.spacing(1)) / (N - 1 + np.spacing(1))).sum() * L

    return score.mean()

def count_distribution(ev, size, use_polarity=True):
    bins_, range_  = [size[0], size[1]], [[0, size[0]], [0, size[1]]]

    if use_polarity:
        weights = (-1) ** (1 + ev[:, 3].astype(np.float_))
    else:
        weights = (+1) ** (0 + ev[:, 3].astype(np.float_))

    counts, *_ = np.histogram2d(ev[:, 1], ev[:, 2], weights=weights, bins=bins_, range=range_)

    return counts


if __name__ == '__main__':
    
    events = np.random.rand(100000, 4)
    x_res = 256
    y_res = 256
    count = 20000
    refN = 20000

    results = run_analysis(events, x_res, y_res, count, refN)

    # plt.show()
    print(results)