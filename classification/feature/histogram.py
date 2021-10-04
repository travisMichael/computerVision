
import numpy as np


def f_histogram(data):
    data_copy = np.copy(data).astype(np.int)
    min = np.min(data_copy)

    if min < 0:
        data_copy = data_copy + min * -1
    max = np.max(data_copy)
    # range=(100, 300), bins=200)
    hist, _ = np.histogram(data_copy, range=(0, max), bins=max-min + 1)
    return hist, min
