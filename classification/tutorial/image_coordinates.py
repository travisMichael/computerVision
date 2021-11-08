from tool import visualize
from tool import image_util as util
import numpy as np
import cv2

kernel_x = np.array([
    [0, 0, 0],
    [-1, 0, 1],
    [0, 0, 0]
])
kernel_y = np.array([
    [0, 1, 0],
    [0, 0, 0],
    [0, -1, 0]
])

ex = np.array([
    [1.0, 4, 2, 6],
    [1, 4, 2, 6]
])


def main():
    # visualize.show_with_handler('../data/experiment/input/lines.jpg')

    file = '../data/experiment/input/lines.jpg'
    img = util.load_grey(file)
    image_f = img.astype(np.float)
    my = 279
    mx = 347 # 266, 355 - 333, 314    18, 3- 0,14
    window = img[my-1:my+4, mx-1:mx+4]

    dx = cv2.filter2D(image_f, -1, kernel_x)
    dy = cv2.filter2D(image_f, -1, kernel_y)
    w = 3

    o = compute_orientation(dx, dy, mx, my, 1)
    # print(o)
    # o = compute_orientation(dx, dy, mx, my, 2)
    # print(o)
    # o = compute_orientation(dx, dy, mx, my, 3)
    # print(o)

    print("done")


def compute_orientation(dx, dy, mx, my, w):
    lambda_, vector_ = compute_dominant_gradiant(dx, dy, mx, my, w)
    v_1 = vector_[:, 0]
    v_2 = vector_[:, 1]
    return np.arctan(v_1[0]/v_1[1]), np.arctan(v_2[0]/v_2[1]), lambda_[0] / lambda_[1]


def compute_dominant_gradiant(dx, dy, mx, my, w):

    dx_w = dx[my-w:my+w+1, mx-w:mx+w+1]
    dy_w = dy[my-w:my+w+1, mx-w:mx+w+1]
    dx_sum = np.sum(dx_w * dx_w)
    dy_sum = np.sum(dy_w * dy_w)
    dx_dy_sum = np.sum(dx_w * dy_w)

    M = np.array([
        [dx_sum, dx_dy_sum],
        [dx_dy_sum, dy_sum]
    ])

    return np.linalg.eig(M)


if __name__ == "__main__":
    main()
