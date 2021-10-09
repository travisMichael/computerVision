import cv2
import numpy as np

from tool import image_util as util
from tool import visualize


def main():
    image = util.load('../../data/experiment/input/lines.jpg')
    b, g, r = cv2.split(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    cv2.imwrite('../../data/experiment/output/h.jpg', h+75)
    cv2.imwrite('../../data/experiment/output/v.jpg', v)
    i = 272
    j = 369
    # visualize.show_with_handler('../../data/experiment/input/lines.jpg')
    print()
    # r 158 - 117
    # g 120 -  83
    # b  79 -  51


if __name__ == "__main__":
    main()
# 186 174
# 222 388
# 754 492