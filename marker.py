import cv2
import numpy as np


def get_algorithm_params_for_shower():
    image = cv2.imread("image_set_2/shower_marker.jpg")

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
    image = cv2.imread("image_set_3/anne_marker.jpg")

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
    image = cv2.imread("image_set_1/island_marker.jpg")

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
