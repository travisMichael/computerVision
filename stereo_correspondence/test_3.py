import cv2

im = cv2.imread("cones_disparity_1_c.png")
cv2.imwrite("cones_disparity_1_c_bright.png", im * 9)

