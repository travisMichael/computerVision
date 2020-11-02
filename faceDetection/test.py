import numpy as np
import cv2

example = cv2.imread("input_images/pos/test0.png")
image = cv2.imread("input_images/man.jpeg")
image = cv2.resize(image, (120, 60))

x = 62
y = 27
t = 12
sub_window = image[y-t:y+t, x-t:x+t]
cv2.imwrite("sub_window_1.png", sub_window)



def do_it():
    for i in range(3):
        for j in range(3):
            x_ = x - i
            y_ = y - j
            sub_window = image[y_-t:y_+t, x_-t:x_+t]
            cv2.imwrite("input_images/pos3/sample_" + str(i) + "_" + str(j) + "plus.png", sub_window)

            sub_window = np.flip(sub_window, axis=1)
            cv2.imwrite("input_images/pos3/sample_" + str(i) + "_" + str(j) + "plus_flip.png", sub_window)

            x_ = x + i
            y_ = y + j
            sub_window = image[y_-t:y_+t, x_-t:x_+t]
            cv2.imwrite("input_images/pos3/sample_" + str(i) + "_" + str(j) + "minus.png", sub_window)

            sub_window = np.flip(sub_window, axis=1)
            cv2.imwrite("input_images/pos3/sample_" + str(i) + "_" + str(j) + "minus_flip.png", sub_window)
    h = image.shape[0]
    w = image.shape[1]
    for i in range(15, h - 15, 5):
        for j in range(20, w - 20, 5):
            x_ = j
            y_ = i
            distance = np.sqrt( (x-x_)**2 + (y-y_)**2)
            if distance < 20:
                continue
            sub_window = image[y_-t:y_+t, x_-t:x_+t]
            cv2.imwrite("input_images/neg3/sample_" + str(i) + "_" + str(j) + "plus.png", sub_window)
            if sub_window.shape[0] != 24 or sub_window.shape[1] != 24:
                print()
            print(sub_window.shape[0], sub_window.shape[1], i, j)

# print("hello")

if __name__ == "__main__":
    do_it()