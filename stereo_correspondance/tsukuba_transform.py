import cv2


def left():
    im = cv2.imread("input_images/tsukuba/scene1.row3.col1.ppm")
    return im


def right():
    im = cv2.imread("input_images/tsukuba/scene1.row3.col3.ppm")
    return im


def ground_truth():
    im = cv2.imread("input_images/tsukuba/truedisp.row3.col3.pgm")
    return im


cv2.imwrite("input_images/tsukuba/right.png", left())
cv2.imwrite("input_images/tsukuba/left.png", right())
cv2.imwrite("input_images/tsukuba/true_disp.png", ground_truth())

print()