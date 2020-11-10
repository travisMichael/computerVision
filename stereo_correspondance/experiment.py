import cv2
import stereo


def test_1():
    left = cv2.imread("input_images/Couch-perfect/im0.png", 0)
    right = cv2.imread("input_images/Couch-perfect/im1.png", 0)

    left = cv2.resize(left, (800, 800))
    right = cv2.resize(right, (800, 800))

    cv2.imwrite("output/left.png", left)
    cv2.imwrite("output/right.png", right)
    disparity_map = stereo.pairwise_stereo(left, right, method="ssd")
    print("hello")


if __name__ == '__main__':
    test_1()
