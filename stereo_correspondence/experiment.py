import cv2
import stereo
import numpy as np
import analyze


def test_1_a():
    left = cv2.imread("input_images/cones/im6.png", 0)
    right = cv2.imread("input_images/cones/im2.png", 0)

    left = cv2.resize(left, (225, 190))
    right = cv2.resize(right, (225, 190))

    cv2.imwrite("output/left.png", left)
    cv2.imwrite("output/right.png", right)
    disparity_map = stereo.pairwise_stereo_ssd(left, right, 7)

    disparity_map[disparity_map > 50] = 0
    disparity_map *= 9

    cv2.imwrite("cones_disparity_1_a.png", disparity_map)
    analyze.analyze_1_a()


def test_1_b():
    left = cv2.imread("input_images/cones/im6.png", 0)
    right = cv2.imread("input_images/cones/im2.png", 0)

    left = cv2.resize(left, (225, 190))
    right = cv2.resize(right, (225, 190))

    cv2.imwrite("output/left.png", left)
    cv2.imwrite("output/right.png", right)
    disparity_map = stereo.pairwise_stereo_dp(left, right, 30)

    disparity_map[disparity_map > 50] = 0
    disparity_map *= 9

    cv2.imwrite("cones_disparity_1_b.png", disparity_map)
    analyze.analyze_1_b()


def test_1_c():
    left = cv2.imread("input_images/cones/im6.png")
    right = cv2.imread("input_images/cones/im2.png")

    left = cv2.resize(left, (225, 190))
    right = cv2.resize(right, (225, 190))
    labels = np.array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26])
    lambda_v = 3
    K = 7
    d_thresh = 150
    full_n = False
    reverse = False
    # labels *= -1

    disparity_map = stereo.pairwise_stereo_graph_cut(left, right, labels, lambda_v, d_thresh, K, full_n, reverse)
    cv2.imwrite("cones_disparity_1_c.png", disparity_map)
    analyze.analyze_1_c()


def test_2_a():
    left = cv2.imread("input_images/tsukuba/scene1.row3.col3.ppm", 0)
    right = cv2.imread("input_images/tsukuba/scene1.row3.col1.ppm", 0)

    disparity_map = stereo.pairwise_stereo_ssd(left, right, 9)

    disparity_map[disparity_map > 50] = 0
    disparity_map *= 9

    cv2.imwrite("tsukuba_disparity_2_a.png", disparity_map)
    analyze.analyze_2_a()


def test_2_b():
    left = cv2.imread("input_images/tsukuba/scene1.row3.col3.ppm", 0)
    right = cv2.imread("input_images/tsukuba/scene1.row3.col1.ppm", 0)

    # disparity_map = stereo.pairwise_stereo_ssd(left, right, 9)
    disparity_map = stereo.pairwise_stereo_dp(left, right, 30)

    disparity_map[disparity_map > 50] = 0
    disparity_map *= 9

    cv2.imwrite("tsukuba_disparity_2_b.png", disparity_map)
    analyze.analyze_2_b()


def test_2_c():
    left = cv2.imread("input_images/tsukuba/scene1.row3.col3.ppm")
    right = cv2.imread("input_images/tsukuba/scene1.row3.col1.ppm")

    left = cv2.resize(left, (192, 144))
    right = cv2.resize(right, (192, 144))
    labels = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    lambda_v = 3
    K = 5
    d_thresh = 100
    full_n = False
    reverse = False
    # labels *= -1

    disparity_map = stereo.pairwise_stereo_graph_cut(left, right, labels, lambda_v, d_thresh, K, full_n, reverse)
    cv2.imwrite("tsukuba_disparity_2_c.png", disparity_map)
    analyze.analyze_2_c()


if __name__ == '__main__':
    test_1_a()
    test_1_b()
    test_1_c()
    test_2_a()
    test_2_b()
    test_2_c()
