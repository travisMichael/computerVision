import cv2
import numpy as np
import matplotlib.pyplot as plt

from feature.gradiant import f_gradiant_convolution, f_magnitude, f_orientation


def main():
    image = cv2.imread('../data/experiment/lines.jpg', cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (600, 800))

    image_f = image.astype(np.float)

    dx, dy = f_gradiant_convolution(image_f)
    magnitude = f_magnitude(dx, dy)
    orientation = f_orientation(dx, dy)

    rng = np.random.RandomState(10)
    a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))

    plt.hist(a, bins='auto')
    plt.show()

    print()

    mx = 268
    my = 354
    # mx-7:mx+7, my-7:my+7
    cv2.imwrite('result.jpg', image)
    # cv2.imshow('T', image)
    # cv2.waitKey(0)
    print()


if __name__ == "__main__":
    main()
