import cv2
import stereo
import numpy as np


def test_1():
    left = cv2.imread("input_images/Couch-perfect/im0.png", 0)
    right = cv2.imread("input_images/Couch-perfect/im1.png", 0)

    left = cv2.resize(left, (800, 800))
    right = cv2.resize(right, (800, 800))

    cv2.imwrite("output/left.png", left)
    cv2.imwrite("output/right.png", right)
    disparity_map = stereo.pairwise_stereo(left, right, None, method="ssd")
    print("hello")


def test_2():
    left = cv2.imread("input_images/Couch-perfect/im1.png", 0)
    right = cv2.imread("input_images/Couch-perfect/im0.png", 0)

    left = cv2.resize(left, (400, 400))
    right = cv2.resize(right, (400, 400))
    labels = np.array([50, 70, 90, 110, 130, 150])

    disparity_map = stereo.pairwise_stereo(left, right, labels, method="graph_cut")
    print("hello")


if __name__ == '__main__':
    # test_1()
    test_2()
