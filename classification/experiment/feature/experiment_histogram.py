import cv2
import numpy as np
import matplotlib.pyplot as plt

from feature.gradiant import f_gradiant_convolution, f_magnitude, f_orientation
from feature.histogram import f_histogram


def main():
    image = cv2.imread('../../data/experiment/input/lines.jpg', cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (600, 800))

    image_f = image.astype(np.float)

    dx, dy = f_gradiant_convolution(image_f)
    magnitude = f_magnitude(dx, dy)
    # orientation = f_orientation(dx, dy)

    # histogram_image, offset = f_histogram(image_f)
    # x = np.arange(histogram_image.shape[0])
    # plt.bar(x, histogram_image, color='maroon')
    # plt.xlabel("Pixel Value")
    # plt.ylabel("Mode")
    # plt.title("Pixel Value Frequency")
    # plt.savefig('../../data/experiment/output/image_hist.png')

    histogram_dx, offset = f_histogram(dx)
    histogram_dx[150:220] = 0
    x = np.arange(histogram_dx.shape[0])
    plt.bar(x, histogram_dx, color='maroon')
    # fig, ax = plt.subplots()
    # ax.set_yscale('log')
    plt.xlabel("dx")
    plt.ylabel("Mode")
    plt.title("dx Frequency")
    plt.savefig('../../data/experiment/output/dx_hist.png')


    print()

    mx = 268
    my = 354
    # mx-7:mx+7, my-7:my+7
    # cv2.imwrite('result.jpg', image)
    # cv2.imshow('T', image)
    # cv2.waitKey(0)
    print()


if __name__ == "__main__":
    main()
