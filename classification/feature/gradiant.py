import numpy as np
import cv2

kernel_r_d = np.array([
    [-1, 0, 0],
    [0, 0, 0],
    [0, 0, 1]
], np.float32)
kernel_r = np.array([
    [0, 0, 0],
    [-1, 0, 1],
    [0, 0, 0]
], np.float32)
kernel_r_u = np.array([
    [0, 0, 1],
    [0, 0, 0],
    [-1, 0, 0]
], np.float32)
kernel_u = np.array([
    [0, 1, 0],
    [0, 0, 0],
    [0, -1, 0]
], np.float32)


# todo: gradiant for all channels
def f_gradiant_convolution(image_f):

    d_r_d = cv2.filter2D(image_f, -1, kernel_r_d)
    d_r = cv2.filter2D(image_f, -1, kernel_r)
    d_r_u = cv2.filter2D(image_f, -1, kernel_r_u)
    d_u = cv2.filter2D(image_f, -1, kernel_u)

    d_r_d_p = np.copy(d_r_d)
    d_r_d_n = np.copy(d_r_d)
    d_r_d_p[d_r_d <= 0] = 0
    d_r_d_n[d_r_d > 0] = 0
    d_r_d_n = d_r_d_n * -1

    d_r_u_p = np.copy(d_r_u)
    d_r_u_n = np.copy(d_r_u)
    d_r_u_p[d_r_u <= 0] = 0
    d_r_u_n[d_r_u > 0] = 0
    d_r_u_n = d_r_u_n * -1

    d_r_d_p = np.sqrt(d_r_d_p)
    d_r_d_n = np.sqrt(d_r_d_n)
    d_r_u_p = np.sqrt(d_r_u_p)
    d_r_u_n = np.sqrt(d_r_u_n)

    dx = d_r + d_r_d_p - d_r_d_n + d_r_u_p - d_r_u_n
    dy = d_u - d_r_d_p + d_r_d_n + d_r_u_p - d_r_u_n
    return dx, dy


def f_gradiant_difference(image_f):
    h, w = image_f.shape
    dy_f = np.diff(image_f, axis=0)
    dx_f = np.diff(image_f, axis=1)

    b_x = np.ones((h, 1))
    dx_f = np.hstack((dx_f, b_x))

    b_y = np.ones((1, w))
    dy_f = np.vstack((dy_f, b_y))
    return dx_f, dy_f


def f_magnitude(dx, dy):
    return np.sqrt(dx * dx + dy * dy)


def f_orientation(dx, dy):
    dx = dx * -1
    dy_p = np.copy(dy)
    dy_n = np.copy(dy)
    dy_p[dy <= 0] = 0
    dy_n[dy > 0] = 0
    theta_y_p = np.where(dy_p != 0, np.arctan(dx/dy_p) + np.pi/2, dy_p)
    theta_y_n = np.where(dy_n != 0, np.arctan(dx/dy_n) + 3*np.pi/2, dy_n)
    return theta_y_p + theta_y_n
