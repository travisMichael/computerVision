import numpy as np

# X is a set of points where X_j is point j.
# Each X_j consists of two components X_j_1 and X_j_2

X = np.array([
    [1.2, -1.0, 1.3],
    [1.1, 1.0, -0.5]
])

M = np.expand_dims(np.mean(X, axis=1), axis=1)
print("Mean Matrix: ", M)

B = X - M
print("X_hat: ", B)

S = np.matmul(B, np.transpose(B))
print("S: ", S)

r = np.linalg.matrix_rank(S)
print("rank of S: ", r)
