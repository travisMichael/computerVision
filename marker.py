import cv2
import numpy as np


# def get_algorithm_params():
# #     marker = "bridge/in/Shower_marker"
# #     image = cv2.imread(marker + ".jpg")
# #     image = cv2.resize(image, (750, 1000))
# #
# #     h, w, _ = image.shape
# #     im_copy = np.copy(image)
# #     m_0 = cv2.medianBlur(image[:, :, 0], 1)
# #     m_1 = cv2.medianBlur(image[:, :, 0], 1)
# #     m_2 = cv2.medianBlur(image[:, :, 2], 1)
# #
# #     zero = im_copy[:, :, 0]
# #     one = im_copy[:, :, 1]
# #     two = im_copy[:, :, 2]
# #
# #     cond = np.logical_and(m_2 > 200, m_1<50)
# #     cond = np.logical_and(cond, m_0 < 50)
# #
# #     zero[np.where(cond)] = 0
# #     one[np.where(cond)] = 0
# #     two[np.where(cond)] = 255
# #
# #     np.save(marker, im_copy)
# #     cv2.imwrite(marker + "2.jpg", im_copy)
# #
# #     Phi = np.zeros((h, w))
# #     Phi[:,:] = 255
# #     Phi[np.where(cond)] = 0
# #
# #     np.save(marker, Phi)
# #     cv2.imwrite("Phi.jpg", Phi)
# #     return image
# get_algorithm_params()

# image = cv2.imread("bridge/in/Shower.jpg")
# image = cv2.resize(image, (750, 1000))
# cv2.imwrite("bridge/in/shower_r.jpg", image)
def get_algorithm_params_for_shower():
    image = cv2.imread("bridge/in/Shower_marker2.jpg")

    h, w, _ = image.shape
    im_copy = np.copy(image)
    m_0 = cv2.medianBlur(image[:, :, 0], 1)
    m_1 = cv2.medianBlur(image[:, :, 0], 1)
    m_2 = cv2.medianBlur(image[:, :, 2], 1)

    zero = im_copy[:, :, 0]
    one = im_copy[:, :, 1]
    two = im_copy[:, :, 2]

    cond = np.logical_and(m_2 > 200, m_1<50)
    cond = np.logical_and(cond, m_0 < 50)

    zero[np.where(cond)] = 0
    one[np.where(cond)] = 0
    two[np.where(cond)] = 255

    Phi = np.zeros((h, w))
    Phi[:,:] = 255
    Phi[np.where(cond)] = 0

    return im_copy, Phi


def get_algorithm_params_for_anne():
    image = cv2.imread("anne/in/anne_marker2.jpg")

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

    Phi = np.zeros((h, w))
    Phi[:,:] = 255
    Phi[np.where(cond)] = 0

    return im_copy, Phi


def get_algorithm_params_for_island():
    image = cv2.imread("island/in/island_marker2.jpg")

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

    Phi = np.zeros((h, w))
    Phi[:,:] = 255
    Phi[np.where(cond)] = 0

    return im_copy, Phi
