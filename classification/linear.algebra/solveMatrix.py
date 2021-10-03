import numpy as np

A = np.array([
    [1, 1],
    [2, -4]
])

b = np.array([3, 0])

x = np.linalg.solve(A, b)

print(x)

b_2 = np.dot(A, x)

print(b_2)

v_1 = np.array([
    [1],
    [2]
])

v_2 = np.array([
    [1],
    [-1]
])

M = np.hstack((v_1, v_2))

print(M)
