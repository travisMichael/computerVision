import numpy as np

A = np.array([
    [1, 0],
    [0, 0]
])

r = np.linalg.matrix_rank(A)

print("Rank = ", r)

A = np.array([
    [1, 0],
    [0, 1]
])

r = np.linalg.matrix_rank(A)

print("Rank = ", r)
