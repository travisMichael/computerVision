import cv2
import numpy as np


def get_image():
    marker = "island/in/island_marker"
    image = cv2.imread(marker + ".jpg")
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

    np.save(marker, Phi)
    cv2.imwrite("Phi.jpg", Phi)
    return image

# get_image()

image = cv2.imread("island/in/island.jpg")
image = cv2.resize(image, (450, 600))
cv2.imwrite("island/in/island_r.jpg", image)