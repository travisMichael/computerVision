import cv2
import numpy as np
# import graph_2 as graph
import graph_3 as g3


def pairwise_stereo_ssd(left, right, t_size):
    h = left.shape[0]
    w = left.shape[1]
    right_expanded = cv2.copyMakeBorder(np.copy(right), t_size, t_size, t_size, t_size, cv2.BORDER_REFLECT)
    left_expanded = cv2.copyMakeBorder(np.copy(left), t_size, t_size, t_size, t_size, cv2.BORDER_REFLECT)
    raw_disparity_map = np.zeros((h, w), dtype=np.float)

    for i in range(h):
        for j in range(t_size, w + t_size):
            left_window = left_expanded[i:i+t_size, j-t_size:j+t_size]
            right_strip = right_expanded[i:i+t_size, :]
            error_map = cv2.matchTemplate(right_strip, left_window, method=cv2.TM_SQDIFF)
            error_map = np.squeeze(error_map)
            error_map[0:j+1] = error_map.max()
            min_arg = np.argmin(error_map)
            raw_disparity_map[i, j - t_size] = min_arg - j + t_size

    for i in range(h):
        for j in range(t_size, w + t_size):
            right_window = right_expanded[i:i+t_size, j-t_size:j+t_size]
            left_strip = left_expanded[i:i+t_size, :]
            error_map = cv2.matchTemplate(left_strip, right_window, method=cv2.TM_SQDIFF)
            error_map = np.squeeze(error_map)
            error_map[j:w+1] = error_map.max()
            q = np.where(error_map == error_map.min())
            if q[0].shape[0] > 1:
                print()
            min_arg = np.argmin(error_map)
            if j - min_arg < 0:
                print()
            raw_disparity_map[i, j - t_size] = j - min_arg - t_size

    return raw_disparity_map


def pairwise_stereo_graph_cut(left, right, labels, lambda_v, d_thresh, K, full_n, reverse):
    stereo_algorithm = g3.AlphaExpansion(left, right, labels, lambda_v=lambda_v,
                                         d_thresh=d_thresh, K=K, full_n=full_n, reverse=reverse)

    f = stereo_algorithm.calculate_disparity_map()

    return f

