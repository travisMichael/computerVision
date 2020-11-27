import cv2
import numpy as np
# import graph_2 as graph
import graph_3 as g3


def pairwise_stereo_ssd(left, right, t_size):
    h = left.shape[0]
    w = left.shape[1]
    # t = window_size
    # t = 3
    right_expanded = cv2.copyMakeBorder(np.copy(right), t_size, t_size, t_size, t_size, cv2.BORDER_REFLECT)
    left_expanded = cv2.copyMakeBorder(np.copy(left), t_size, t_size, t_size, t_size, cv2.BORDER_REFLECT)
    raw_disparity_map = np.zeros((h, w), dtype=np.float)

    for i in range(h):
        if i % 20 == 0:
            print(i)
        for j in range(t_size, w + t_size):
            left_window = left_expanded[i:i+t_size, j-t_size:j+t_size]
            right_strip = right_expanded[i:i+t_size, :]
            # cv2.imwrite("output/left_window.png", left_window)
            # cv2.imwrite("output/right_strip.png", right_strip)
            # print(j)
            error_map = cv2.matchTemplate(right_strip, left_window, method=cv2.TM_SQDIFF)
            error_map = np.squeeze(error_map)
            error_map[0:j+1] = error_map.max()
            min_arg = np.argmin(error_map)
            # if min_arg - j <= 0:
            #     print("Not good")
            raw_disparity_map[i, j - t_size] = min_arg - j + t_size

    return raw_disparity_map


def pairwise_stereo_graph_cut(left, right, labels, lambda_v, d_thresh, K, full_n, reverse):
    stereo_algorithm = g3.AlphaExpansion(left, right, labels, lambda_v=lambda_v,
                                         d_thresh=d_thresh, K=K, full_n=full_n, reverse=reverse)

    f = stereo_algorithm.calculate_disparity_map()

    return f

