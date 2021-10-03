import numpy as np

# perspective invariant transform

# what is it?

# a feature can be described as a set of points that "describe" the feature

# an observer should be able to label a feature if the observer perceives the feature at
#   different locations.

# **All features have a base representation.

# PIT will transform a feature, f, into its base representation, b. Pit will also transform any linear
#    transformation of the feature, f, into the same base representation

# Demonstration is in a 2 dimensional point class space


# this is a crude implementation..
def transform_2d(X):

    v_1 = X[:, 1] - X[:, 0]
    v_2 = X[:, 3] - X[:, 0]
    v_1 = np.expand_dims(v_1, axis=1)
    v_2 = np.expand_dims(v_2, axis=1)

    M = np.hstack((v_1, v_2))

    c = np.linalg.solve(M, X[:, 2])
    return c


if __name__ == "__main__":

    X = np.array([
        [-1.0, 0.0, 3.0,  0.0],
        [ 0.0, 2.0, 0.0, -4.0]
    ])

    result_1 = transform_2d(X)
    print(result_1)

    A = np.array([
        [2.0, 0.0],
        [0.0, 2.0]
    ])

    X_2 = np.matmul(A, X)
    result_2 = transform_2d(X)
    print(result_2)




# normalized_diff_v = diff_v / np.linalg.norm(diff_v, axis=0)
#
# r = np.dot(np.transpose(normalized_diff_v), normalized_diff_v)
# r_abs = np.abs(r)
# # we only care about the lower half
# r_abs[np.tril_indices(3)] = 1
# i = np.unravel_index(np.argmin(r_abs, axis=None), (3, 3))
