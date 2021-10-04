import cv2
import numpy as np
from feature.gradiant import f_gradiant_convolution, f_magnitude, f_orientation


def main():
    image = cv2.imread('../../data/experiment/input/lines.jpg', cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (600, 800))

    image_f = image.astype(np.float)

    dx, dy = f_gradiant_convolution(image_f)
    magnitude = f_magnitude(dx, dy)
    # todo fix orientation
    # orientation = f_orientation(dx, dy)

    mx = 268
    my = 354
    # mx-7:mx+7, my-7:my+7
    cv2.imwrite('result.jpg', image)
    # cv2.imshow('T', image)
    # cv2.waitKey(0)
    print()


if __name__ == "__main__":
    main()
