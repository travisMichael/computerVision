import numpy as np


def initialize_labeling_function(size, number_of_labels):
    f = np.random.randint(low=0, high=number_of_labels, size=size)

    return f


def calculate_energy(f, L, R):
    s = calculate_smoothness_term(f, L, R)
    d = calculate_data_term(f, L, R)

    return d + s


def D_p(p, label, L, R):
    h = L.shape[0]
    w = L.shape[1]
    THRESHOLD = 20
    increment = 2
    p_index = np.unravel_index(p, (h, w))

    I_p = L[p_index]

    left = p_index[1] + np.max([label - increment, 0])
    right = p_index[1] + np.min([label + increment, w - 1])

    if left > w - 1:
        return THRESHOLD

    pixel_values = R[p_index[0], left:right]

    abs_diff = abs(pixel_values - I_p)
    value = np.min(abs_diff)
    if value > THRESHOLD:
        return THRESHOLD

    return value


def calculate_data_term(f, L, R):
    h = L.shape[0]
    w = L.shape[1]
    pixel = 0
    sum = 0.0
    for i in range(h):
        # if i % 50 == 0:
        #     print(i)
        for j in range(w):
            sum += D_p(pixel, f[pixel], L, R)
            pixel += 1

    return sum


def calculate_neighborhood_term(p, q, f, L, R):
    p_label = f[p]
    q_label = f[p]

    h = L.shape[0]
    w = L.shape[1]

    if p_label == q_label:
        return 0.0

    p_index = np.unravel_index(p, (h, w))
    q_index = np.unravel_index(q, (h, w))

    term_2 = L[q_index[0], q_index[1], :]

    intensity_diff = np.sqrt(np.sum((L[p_index[0], p_index[0], :] - term_2) ** 2))

    if intensity_diff > 65:
        return 0

    # 25 for couch
    return 0


def calculate_smoothness_term(f, L, R):
    h = L.shape[0]
    w = L.shape[1]
    pixel = 0
    sum = 0.0
    for i in range(h):
        for j in range(w):

            if i < h - 1:
                bottom_pixel = pixel + w
                sum += calculate_neighborhood_term(pixel, bottom_pixel, f, L, R)

            if j < w - 1:
                right_pixel = pixel + 1
                sum += calculate_neighborhood_term(pixel, right_pixel, f, L, R)
            pixel += 1

    return sum