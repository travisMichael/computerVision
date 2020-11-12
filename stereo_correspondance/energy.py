import numpy as np


def initialize_labeling_function(size, number_of_labels):
    f = np.random.randint(low=0, high=number_of_labels, size=size)

    return f


def calculate_energy(f, left, right):

    return 0.0
