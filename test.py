import numpy as np
import cv2

image = np.array([
    [0, 1, 2, 10, 4, 5],
    [6, 7, 17, 9, 10, 11],
    [12, 2, 14, 20, 16, 17],
    [18, 19, 20, 21, 22, 23],
    [18, 19, 20, 21, 22, 23],
]).astype("uint8")

patch = np.array([
    [0, 1],
    [6, 7],
    [12, 2]
]).astype("uint8")

res_0 = cv2.matchTemplate(image, patch, method=cv2.TM_SQDIFF)

print(res_0)
# print(a)
# print(b)