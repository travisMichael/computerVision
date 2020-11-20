import cv2
import stereo
import numpy as np


def test_1():
    left = cv2.imread("input_images/Couch-perfect/im1.png", 0)
    right = cv2.imread("input_images/Couch-perfect/im0.png", 0)

    left = cv2.resize(left, (200, 200))
    right = cv2.resize(right, (200, 200))

    cv2.imwrite("output/left.png", left)
    cv2.imwrite("output/right.png", right)
    disparity_map = stereo.pairwise_stereo(left, right, None, method="ssd")
    print("hello")


def test_2():
    left = cv2.imread("input_images/Couch-perfect/im1.png")
    right = cv2.imread("input_images/Couch-perfect/im0.png")

    left = cv2.resize(left, (200, 200))
    right = cv2.resize(right, (200, 200))
    # labels = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, ])
    # labels = np.array([6, 12, 18, 24, 30, 36, 42, 48, 54])
    labels = np.array([5, 13, 20, 28, 35, 43, 50, 57])

    disparity_map = stereo.pairwise_stereo(left, right, labels, method="graph_cut")
    print("hello")


def test_2_a():
    left = cv2.imread("input_images/cones/im6.png", 0)
    right = cv2.imread("input_images/cones/im2.png", 0)

    left = cv2.resize(left, (200, 200))
    right = cv2.resize(right, (200, 200))

    cv2.imwrite("output/left.png", left)
    cv2.imwrite("output/right.png", right)
    disparity_map = stereo.pairwise_stereo(left, right, None, method="ssd")
    print("hello")


# def test_2_b():
#     left = cv2.imread("input_images/cones/im6.png")
#     right = cv2.imread("input_images/cones/im2.png")
#
#     left = cv2.resize(left, (225, 190))
#     right = cv2.resize(right, (225, 190))
#     # labels = np.array([10, 20, 30, 40, 50, 60])
#     # labels = np.array([9, 14, 19, 24, 29, 34, 40])
#     # labels = np.array([9, 12, 15, 18, 21, 24, 27, 30])
#     labels = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27])
#     # labels = np.array([9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29])
#     # labels = np.array([8, 13, 17, 21, 25, 29])
#     # labels = np.array([10, 20, 30, 40, 50, 60, 70])
#     increment = 1
#     k = 0.5
#     k_not = 0.5
#     intensity_thresh = 20
#     d_thresh = 20
#
#     disparity_map = stereo.pairwise_stereo_graph_cut(left, right, labels, increment, k, k_not, intensity_thresh, d_thresh)
#     print("hello")

def test_2_b():
    left = cv2.imread("input_images/cones/im6.png", 0)
    right = cv2.imread("input_images/cones/im2.png", 0)

    left = cv2.resize(left, (225, 190))
    right = cv2.resize(right, (225, 190))
    # labels = np.array([10, 20, 30, 40, 50, 60])
    # labels = np.array([9, 14, 19, 24, 29, 34, 40])
    # labels = np.array([9, 12, 15, 18, 21, 24, 27, 30])
    labels = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27])
    # labels = np.array([9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29])
    # labels = np.array([8, 13, 17, 21, 25, 29])
    # labels = np.array([10, 20, 30, 40, 50, 60, 70])
    lambda_v = 10
    d_thresh = 80

    disparity_map = stereo.pairwise_stereo_graph_cut_3(left, right, labels, lambda_v, d_thresh)
    print("hello")


if __name__ == '__main__':
    # test_1()
    # test_2()
    # test_2_a()
    test_2_b()
