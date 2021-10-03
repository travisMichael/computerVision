import numpy as np

# transform a vector
# doubles the length of a vector
A = np.array([
    [2, 0],
    [0, 2]
])

x = np.array([
    [1],
    [2]
])

b = np.matmul(A, x)

print(b)


# transform a set of vector
# doubles the length of each vector
A = np.array([
    [2, 0],
    [0, 2]
])

x = np.array([
    [1, -3],
    [2, 5]
])

b = np.matmul(A, x)

print(b)
