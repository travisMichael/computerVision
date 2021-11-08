import cv2


def load(file, height=800):
    img = cv2.imread(file)
    h, w = img.shape[:2]
    ratio = w/float(h)
    new_h = height
    new_w = int(new_h*ratio)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def load_grey(file, height=800):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    ratio = w/float(h)
    new_h = height
    new_w = int(new_h*ratio)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img