import cv2
import numpy as np


def compare_with_ground_truth(ground_truth, result, ratio, match):
    h = ground_truth.shape[0]
    w = ground_truth.shape[1]
    ground_truth *= ratio
    thresh = 2
    diff = np.abs(ground_truth - result)
    matching_disparities = np.where(diff < thresh)[0].shape[0]

    p = np.zeros((h,w))
    p[diff < thresh] = 255
    cv2.imwrite(match + ".png", p)

    ratio = float(matching_disparities) / float(h*w)

    return ratio


def analyze_1_a():
    ground_truth = cv2.imread("input_images/cones/disp6.png", 0).astype(np.float) / 4.0
    ssd_result = cv2.imread("cones_disparity_1_a.png", 0).astype(np.float) / 9.0

    ground_truth = cv2.resize(ground_truth, (225, 190))

    ground_truth = ground_truth[:, 0:200]
    ssd_result = ssd_result[:, 0:200]

    ratio_1 = compare_with_ground_truth(ground_truth, ssd_result, 1.0/2.0, "1_a_comparision")
    print(ratio_1)


def analyze_1_b():
    ground_truth = cv2.imread("input_images/cones/disp6.png", 0).astype(np.float) / 4.0
    result = cv2.imread("cones_disparity_1_b.png", 0).astype(np.float) / 9.0

    ground_truth = cv2.resize(ground_truth, (225, 190))

    ground_truth = ground_truth[:, 0:200]
    result = result[:, 0:200]

    ratio_1 = compare_with_ground_truth(ground_truth, result, 1.0/2.0, "1_b_comparision")
    print(ratio_1)


def analyze_1_c():
    ground_truth = cv2.imread("input_images/cones/disp6.png", 0).astype(np.float) / 4.0
    result = cv2.imread("d_21.png", 0).astype(np.float) / 9.0
    ground_truth = cv2.resize(ground_truth, (225, 190))

    ground_truth = ground_truth[:, 0:200]
    result = result[:, 0:200]

    ratio_1 = compare_with_ground_truth(ground_truth, result, 1.0/2.0, "1_c_comparision")
    print(ratio_1)


def analyze_2_a():
    ground_truth = cv2.imread("input_images/tsukuba/truedisp.row3.col3.pgm", 0).astype(np.float) / 8.0
    result = cv2.imread("output/disparity_ssd_map_3_a.png", 0).astype(np.float) / 9.0

    h, w = ground_truth.shape

    ground_truth = ground_truth[17:h-17, 17:w-17]
    result = result[17:h-17, 17:w-17]

    ratio_1 = compare_with_ground_truth(ground_truth, result, 1.0, "2_a_comparision")
    print(ratio_1)


def analyze_2_b():
    ground_truth = cv2.imread("input_images/tsukuba/truedisp.row3.col3.pgm", 0).astype(np.float) / 8.0
    result = cv2.imread("tsukuba_disparity_2_b.png", 0).astype(np.float) / 9.0

    h, w = ground_truth.shape

    ground_truth = ground_truth[17:h-17, 17:w-17]
    result = result[17:h-17, 17:w-17]

    ratio_1 = compare_with_ground_truth(ground_truth, result, 1.0, "2_b_comparision")
    print(ratio_1)


def analyze_2_c():
    ground_truth = cv2.imread("input_images/tsukuba/truedisp.row3.col3.pgm", 0).astype(np.float) / 8.0
    result = cv2.imread("Tsukuba_disparity_2_c.png", 0).astype(np.float) / 9

    ground_truth = cv2.resize(ground_truth, (192, 144))

    h, w = ground_truth.shape

    ground_truth = ground_truth[17:h-17, 17:w-17]
    result = result[17:h-17, 17:w-17]

    ratio_1 = compare_with_ground_truth(ground_truth, result, 1.0/2.0, "2_c_comparision")
    print(ratio_1)


if __name__ == '__main__':
    # analyze_1_a()
    analyze_1_b()
    # analyze_1_c()
    # analyze_2_a()
    # analyze_2_b()
    # analyze_2_b()
