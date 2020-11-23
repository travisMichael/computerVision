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
    disparity_map = stereo.pairwise_stereo_ssd(left, right, 3)

    disparity_map[disparity_map > 60] = 0
    disparity_map *= 4

    cv2.imwrite("output/disparity_ssd_map_1_a.png", disparity_map)


def test_1_b():
    left = cv2.imread("input_images/Couch-perfect/im1.png")
    right = cv2.imread("input_images/Couch-perfect/im0.png")

    left = cv2.resize(left, (200, 200))
    right = cv2.resize(right, (200, 200))
    # labels = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, ])
    # labels = np.array([6, 12, 18, 24, 30, 36, 42, 48, 54])
    labels = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                       20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                       30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                       40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
    lambda_v = 5
    d_thresh = 80

    disparity_map = stereo.pairwise_stereo_graph_cut(left, right, labels, lambda_v, d_thresh)
    print("hello")


def test_2_a():
    left = cv2.imread("input_images/cones/im6.png", 0)
    right = cv2.imread("input_images/cones/im2.png", 0)

    left = cv2.resize(left, (225, 190))
    right = cv2.resize(right, (225, 190))

    cv2.imwrite("output/left.png", left)
    cv2.imwrite("output/right.png", right)
    disparity_map = stereo.pairwise_stereo_ssd(left, right, 7)

    disparity_map[disparity_map > 28] = 0
    disparity_map *= 9

    cv2.imwrite("output/disparity_ssd_map_2_a.png", disparity_map)


def test_2_b():
    left = cv2.imread("input_images/cones/im6.png")
    right = cv2.imread("input_images/cones/im2.png")
    print(left.shape)

    left = cv2.resize(left, (225, 190))
    right = cv2.resize(right, (225, 190))
    labels = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26])
    lambda_v = 3
    d_thresh = 80

    disparity_map = stereo.pairwise_stereo_graph_cut(left, right, labels, lambda_v, d_thresh)
    cv2.imwrite("output/disparity_ssd_map_2_b.png", disparity_map)


def test_3_b():
    left = cv2.imread("input_images/tsukuba/scene1.row3.col3.ppm")
    right = cv2.imread("input_images/tsukuba/scene1.row3.col1.ppm")
    print(left.shape)

    # left = cv2.resize(left, (225, 190))
    # right = cv2.resize(right, (225, 190))
    labels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20])
    lambda_v = 2
    d_thresh = 80

    disparity_map = stereo.pairwise_stereo_graph_cut(left, right, labels, lambda_v, d_thresh)
    cv2.imwrite("output/disparity_ssd_map_3_b.png", disparity_map)


def t():
    d = cv2.imread("output/disparity_ssd_map_2_b.png")
    d[d > 28] = 0
    d *= 9
    cv2.imwrite("d_6.png",d)


    print()

if __name__ == '__main__':
    # t()
    # test_1()
    # test_1_b()
    # test_2_a()
    # test_2_b()
    test_3_b()
