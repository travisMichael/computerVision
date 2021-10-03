import numpy as np

# A should be symmetric
A = np.array([
    [1.6, -1.0],
    [-1.0, 1.6]
])

w, v = np.linalg.eig(A)

print(w)
print("-------")
print(v)
