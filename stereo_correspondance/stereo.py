import cv2
import numpy as np
import graph


# def pairwise_stereo(left, right, labels, method):
#     if method == "ssd":
#         print("calculating stereo map ssd")
#         return pairwise_stereo_ssd(left, right)
#     elif method == "graph_cut":
#         print("calculating stereo map graph cut")
#         return pairwise_stereo_graph_cut(left, right, labels)


def pairwise_stereo_ssd(left, right):
    h = left.shape[0]
    w = left.shape[1]
    # t = window_size
    t = 3
    right_expanded = cv2.copyMakeBorder(np.copy(right), t, t, t, t, cv2.BORDER_REFLECT)
    left_expanded = cv2.copyMakeBorder(np.copy(left), t, t, t, t, cv2.BORDER_REFLECT)
    raw_disparity_map = np.zeros((h, w-40), dtype=np.float)

    for i in range(h):
        if i % 20 == 0:
            print(i)
        for j in range(20, w-20):
            left_window = left_expanded[i:i+t, j-t:j+t]
            right_strip = right_expanded[i:i+t, :]
            # cv2.imwrite("output/left_window.png", left_window)
            # cv2.imwrite("output/right_strip.png", right_strip)
            error_map = cv2.matchTemplate(right_strip, left_window, method=cv2.TM_SQDIFF)
            error_map = np.squeeze(error_map)
            error_map[0:j+1] = error_map.max()
            min_arg = np.argmin(error_map)
            if min_arg - j <= 0:
                print("Not good")
            raw_disparity_map[i, j-20] = min_arg - j
            # print()

    disparity_ssd_map = raw_disparity_map / raw_disparity_map.max() * 255
    cv2.imwrite("output/disparity_ssd_map.png", disparity_ssd_map)
    return None


def pairwise_stereo_graph_cut(left, right, labels, increment, k, k_not, intensity_thresh, d_thresh):

    # use alpha beta swap to get f, where f: pixel --> label (f assigns a label to each pixel).
    stereo_solver = graph.AlphaExpansion(left, right, labels,
                                         increment=increment, k=k, k_not=k_not,
                                         v_thresh=intensity_thresh, d_thresh=d_thresh)

    f = stereo_solver.calculate_disparity_map()
    # transform f into disparity map

    return None

