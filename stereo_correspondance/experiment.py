import cv2
import stereo


def test_1():
    left = cv2.imread("input_images/Couch-perfect/im0.png")
    right = cv2.imread("input_images/Couch-perfect/im1.png")
    disparity_map = stereo.pairwise_stereo(left, right)
    print("hello")


if __name__ == '__main__':
    test_1()
