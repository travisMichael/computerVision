import numpy as np

# X is a set of points where X_j is point j.
# Each X_j consists of two components X_j_1 and X_j_2

# X = np.array([
#     [0.0, 4.0, 0],
#     [0.0, 0, 4.0]
# ])

# X = np.array([
#     [-1.0, 1.0, 0, 0],
#     [0.0, 0, 1.0, -1.0]
# ])

X = np.array([
    [-4.0, 4.0, 0.0,  0.0, 2.0],
    [ 0.0, 0.0, 4.0, -4.0, 2.0]
])

M = np.expand_dims(np.mean(X, axis=1), axis=1)
print("Mean Matrix: \n", M, "\n")

B = X - M
print("X_hat: \n", B, "\n")

S = np.matmul(B, np.transpose(B))
print("S: \n", S, "\n")

r = np.linalg.matrix_rank(S)
print("rank of S: \n", r, "\n")

w, v = np.linalg.eig(S)
print("eigen vectors of S: \n", v, "\n")

print("eigen values of S: \n", w, "\n")
