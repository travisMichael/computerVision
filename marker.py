import cv2
import numpy as np


def get_image():
    marker = "bridge/in/bridge_marker"
    image = cv2.imread(marker + ".jpg")
    # image = cv2.resize(image, (450, 600))
    # cv2.circle(image, (348, 180), 2, [0, 0, 255], thickness=2)

    # h, w, _ = image.shape
    # im_copy = np.copy(image)

    # image = cv2.resize(image, (600, 800))
    # image = image[0:780, :, :]

    h, w, _ = image.shape
    im_copy = np.copy(image)
    m_0 = cv2.medianBlur(image[:, :, 0], 9)
    m_1 = cv2.medianBlur(image[:, :, 0], 9)
    m_2 = cv2.medianBlur(image[:, :, 2], 9)

    zero = im_copy[:, :, 0]
    one = im_copy[:, :, 1]
    two = im_copy[:, :, 2]

    cond = np.logical_and(m_2 > 200, m_1<50)
    cond = np.logical_and(cond, m_0 < 50)

    zero[np.where(cond)] = 0
    one[np.where(cond)] = 0
    two[np.where(cond)] = 255


    np.save(marker, im_copy)
    cv2.imwrite(marker + "2.jpg", im_copy)

    Phi = np.zeros((h, w))
    Phi[:,:] = 255
    Phi[np.where(cond)] = 0

    # for i in range(10, h-10):
    #     for j in range(10, w-10):
    #         m = np.median(image[i-5:i+5, j-5:j+5, 2])
    #         if m > 245:
    #             im_copy[i, j, :] = np.array([0, 0, 255])
    # np.save(marker, im_copy)
    # cv2.imwrite(marker + ".jpg", im_copy)
    #
    # channel_red = im_copy[:,:,2]
    # Phi = np.zeros((h, w))
    # Phi[np.where(channel_red < 230)] = 255
    #
    # for i in range(10, h-10):
    #     for j in range(10, w-10):
    #         m = np.median(im_copy[i-7:i+7, j-7:j+7])
    #         if m > 170:
    #             Phi[i, j] = 255
    #
    # Phi[0:150, :] = 255
    # Phi[300-80:300, :] = 255
    # Phi[:, 0:110] = 255
    # Phi[:, 295-70:300] = 255
    np.save(marker, Phi)
    cv2.imwrite("Phi.jpg", Phi)
    return image

get_image()

# image = cv2.imread("bridge/in/Bridge_marker.jpg")
# # image = cv2.resize(image, (450, 600))
# cv2.imwrite("bridge/in/bridge_marker.jpg", image)