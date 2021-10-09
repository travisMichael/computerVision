import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from tool import image_util as util


def mouse_click_handler(img):
    i = img

    def mouse_click(event, x, y, flags, param):
        l = i
        if event == cv2.EVENT_LBUTTONDOWN:
            # l = i
            # font = cv2.FONT_HERSHEY_TRIPLEX
            # LB = 'Left Button'

            # cv2.putText(img, LB, (x, y), font, 1, (255, 255, 0),2)
            # cv2.imshow('image', img)
            print(y, x)

    return mouse_click


def show_with_handler(file, handler=mouse_click_handler):
    img = util.load(file)

    cv2.imshow("image", img)
    cv2.setMouseCallback('image', handler(img))

    cv2.waitKey(0)

    cv2.destroyAllWindows()


def plot_3d(image):
    r, g, b = cv2.split(image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = image.reshape((np.shape(image)[0]*np.shape(image)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    #
    axis.scatter(r.flatten(), g.flatten(), b.flatten(),  facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()
    cv2.waitKey(0)
