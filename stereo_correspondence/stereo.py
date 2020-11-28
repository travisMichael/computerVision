import cv2
import numpy as np
# import graph_2 as graph
import graph_3 as g3
NO_OP = 1000000000


def calculate_energy(dsi, max_d):
    energy = np.ones_like(dsi) * NO_OP
    w = dsi.shape[0]
    energy[w-1, :] = np.copy(dsi[w-1, :])
    energy[:, w-1] = np.copy(dsi[:, w-1])

    for i in range(w-2, -1, -1):
        for j in range(i, i-max_d, -1):
            if j < 0:
                continue
            right = 100000000
            bottom = 100000000
            bottom_right = 100000000
            if energy[i, j+1] != NO_OP:
                right = energy[i, j+1]
            if energy[i+1, j] != NO_OP:
                bottom = energy[i+1, j]
            if energy[i+1, j+1] != NO_OP:
                bottom_right = energy[i+1, j+1]
            value = np.min([right, bottom, bottom_right])
            energy[i, j] = value + dsi[i, j]

    energy[w-1, 0:w-max_d] = NO_OP
    return energy


def extract_path(energy):
    w = energy.shape[0]
    line_disparity = np.zeros(w)
    j = 0
    i = np.argmin(energy[:, j])
    line_disparity[j] = i - j

    is_done = False
    while not is_done:
        right = 100000000
        bottom = 100000000
        bottom_right = 100000000
        # print(j, i)
        if energy[i, j+1] != NO_OP:
            right = energy[i, j+1]
        if energy[i+1, j] != NO_OP:
            bottom = energy[i+1, j]
        if energy[i+1, j+1] != NO_OP:
            bottom_right = energy[i+1, j+1]
        arg_min = np.argmin([right, bottom, bottom_right])
        if arg_min == 2 and j >= 0:
            line_disparity[j] = j - i
            j = j + 1
            i = i + 1
            pass
        elif arg_min == 1 and j >= 0:
            i = i + 1
            pass
        else:
            line_disparity[j] = j - i
            j = j + 1
            pass

        if i >= w-1:
            break

    return line_disparity*-1


def calculate_dsi(left_scan_line, right_scan_line, w):
    dsi = np.zeros((w, w))

    right_scan_line = np.expand_dims(right_scan_line, axis=1)
    dsi += right_scan_line
    dsi = np.abs(dsi - left_scan_line)
    return dsi


def pairwise_stereo_dp(left, right, max_d):
    h = left.shape[0]
    w = left.shape[1]

    f = np.zeros((h,w))
    for i in range(h):
        left_scan_line = left[i, :]
        right_scan_line = right[i, :]
        dsi = calculate_dsi(left_scan_line, right_scan_line, w)
        energy = calculate_energy(dsi, max_d)
        path = extract_path(energy)
        f[i, :] = path
        # cv2.imwrite("output/dsi.png", dsi + 255 - dsi.max())
        # print(i)

    return f


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

    # for i in range(h):
    #     for j in range(t_size, w + t_size):
    #         right_window = right_expanded[i:i+t_size, j-t_size:j+t_size]
    #         left_strip = left_expanded[i:i+t_size, :]
    #         error_map = cv2.matchTemplate(left_strip, right_window, method=cv2.TM_SQDIFF)
    #         error_map = np.squeeze(error_map)
    #         error_map[j:w+1] = error_map.max()
    #         q = np.where(error_map == error_map.min())
    #         if q[0].shape[0] > 1:
    #             print()
    #         min_arg = np.argmin(error_map)
    #         if j - min_arg < 0:
    #             print()
    #         raw_disparity_map[i, j - t_size] = j - min_arg - t_size

    return raw_disparity_map


def pairwise_stereo_graph_cut(left, right, labels, lambda_v, d_thresh, K, full_n, reverse):
    stereo_algorithm = g3.AlphaExpansion(left, right, labels, lambda_v=lambda_v,
                                         d_thresh=d_thresh, K=K, full_n=full_n, reverse=reverse)

    f = stereo_algorithm.calculate_disparity_map()

    return f

