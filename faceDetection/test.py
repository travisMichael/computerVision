import numpy as np
import cv2


def generate_examples():
    image = cv2.imread("input_images/man.jpeg")
    image = cv2.resize(image, (120, 60))

    x = 62
    y = 27
    t = 12

    pos = []
    neg = []
    for i in range(5):
        for j in range(5):
            x_ = x - i
            y_ = y - j
            sub_window = np.copy(image[y_-t:y_+t, x_-t:x_+t])
            pos.append(cv2.cvtColor(sub_window, cv2.COLOR_BGR2GRAY))

            x_ = x + i
            y_ = y + j
            sub_window = np.copy(image[y_-t:y_+t, x_-t:x_+t])
            pos.append(cv2.cvtColor(sub_window, cv2.COLOR_BGR2GRAY))

    h = image.shape[0]
    w = image.shape[1]
    for i in range(15, h - 15, 5):
        for j in range(20, w - 20, 5):
            x_ = j
            y_ = i
            distance = np.sqrt( (x-x_)**2 + (y-y_)**2)
            if distance < 20:
                continue
            sub_window = np.copy(image[y_-t:y_+t, x_-t:x_+t])
            neg.append(cv2.cvtColor(sub_window, cv2.COLOR_BGR2GRAY))

    return pos, neg


if __name__ == "__main__":
    generate_examples()
