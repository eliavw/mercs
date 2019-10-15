import numpy as np

# Helpers
def _init_thresholds(init_threshold, stepsize):
    """Initialize thresholds array based on its two defining parameters.

    Parameters
    ----------
    init_threshold: 
    stepsize: 

    Returns
    -------

    """

    thresholds = np.arange(init_threshold, -1 - stepsize, -stepsize)
    thresholds = np.clip(thresholds, -1, 1)
    return thresholds