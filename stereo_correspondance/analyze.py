import cv2
import numpy as np


def compare_with_ground_truth(ground_truth, result, ratio, match):
    h = ground_truth.shape[0]
    w = ground_truth.shape[1]
    ground_truth *= ratio

    diff = np.abs(ground_truth - result)

    matching_disparities = np.where(diff <= 1.9)[0].shape[0]

    p = np.zeros((h,w))
    p[diff <= 1.9] = 255
    cv2.imwrite("output/" + match + ".png", p)

    ratio = float(matching_disparities) / float(h*w)

    return ratio


def analyze_2_a():
    ground_truth = cv2.imread("input_images/cones/disp6.png", 0).astype(np.float) / 4.0
    ssd_result = cv2.imread("output/disparity_ssd_map_2_a.png", 0).astype(np.float) / 9.0

    ground_truth = cv2.resize(ground_truth, (225, 190))

    ground_truth = ground_truth[:, 0:200]
    ssd_result = ssd_result[:, 0:200]

    ratio_1 = compare_with_ground_truth(ground_truth, ssd_result, 1.0/2.0, "match_2_a")
    print(ratio_1)


def analyze_2_b():
    ground_truth = cv2.imread("input_images/cones/disp6.png", 0).astype(np.float) / 4.0
    # ssd_result = cv2.imread("output/disparity/d_27.png", 0).astype(np.float) / 9.0
    ssd_result = cv2.imread("d_21.png", 0).astype(np.float) / 9.0
    ground_truth = cv2.resize(ground_truth, (225, 190))

    ground_truth = ground_truth[:, 0:200]
    ssd_result = ssd_result[:, 0:200]

    ratio_1 = compare_with_ground_truth(ground_truth, ssd_result, 1.0/2.0, "match_2_b")
    print(ratio_1)


def analyze_3_b():
    ground_truth = cv2.imread("input_images/tsukuba/truedisp.row3.col3.pgm", 0).astype(np.float) / 8.0
    # ssd_result = cv2.imread("output/disparity/d_27.png", 0).astype(np.float) / 9.0
    result = cv2.imread("output/disparity_ssd_map_3_b.png", 0).astype(np.float)
    ground_truth = cv2.resize(ground_truth, (192, 144))

    h, w = ground_truth.shape

    ground_truth = ground_truth[17:h-17, 17:w-17]
    result = result[17:h-17, 17:w-17]
    # ground_truth = ground_truth[:, 0:200]
    # ssd_result = ssd_result[:, 0:200]

    ratio_1 = compare_with_ground_truth(ground_truth, result, 1.0/2.0, "match_2_b")
    print(ratio_1)


if __name__ == '__main__':
    # analyze_2_a()
    analyze_3_b()
