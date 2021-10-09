import cv2


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
    img = cv2.imread(file)
    h, w = img.shape[:2]
    ratio = w/float(h)
    new_h = 800
    new_w = int(new_h*ratio)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.imshow("image", img)
    cv2.setMouseCallback('image', handler(img))

    cv2.waitKey(0)

    cv2.destroyAllWindows()