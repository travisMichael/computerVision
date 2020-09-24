"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np

# CORNER_KERNEL = np.array([
#     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
# ]).astype(np.float)

CORNER_KERNEL = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
]).astype(np.float)

DIAGONAL_CORNER_KERNEL = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]).astype(np.float)

DIAGONAL_CORNER_KERNEL[DIAGONAL_CORNER_KERNEL == 1] = -1
DIAGONAL_CORNER_KERNEL[DIAGONAL_CORNER_KERNEL == 0] = 1


SPEC_KERNEL = np.array([
    [-1, -1, -1],
    [-1, 1, -1],
    [-1, -1, -1],
]).astype(np.float)

r = np.zeros((32, 32, 3))
r[0:16, 0:16] = [255, 255, 255]
r[16:,16:] = [255, 255, 255]

d = np.zeros((32, 32, 3))
d[16:, 0:16] = [255, 255, 255]
d[0:16, 16:] = [255, 255, 255]

s = np.zeros((32, 32, 3))

for i in range(32):
    for j in range(32):
        if i < 16:
            if j < 16:
                if i <= j:
                    s[i, j] = [255, 255, 255]
            else:
                if j+i < 31:
                    s[i, j] = [255, 255, 255]
        else:
            if j < 16:
                if j + i >= 31:
                    s[i, j] = [255, 255, 255]
            else:
                if i > j:
                    s[i, j] = [255, 255, 255]

t = np.copy(s)
t[s == 255] = 0
t[s == 0] = 255


def rotate_template_45(im):
    h, w, _ = im.shape
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for i in range(map_x.shape[0]):
        map_x[i,:] = [x for x in range(map_x.shape[1])]
    for j in range(map_y.shape[1]):
        map_y[:,j] = [y for y in range(map_y.shape[0])]

    map_x_copy = np.copy(map_x)
    map_y_copy = np.copy(map_y)

    theta = np.pi / 4

    for i in range(map_x.shape[0]):
        for j in range(map_x.shape[1]):
            map_x_copy[i, j] = map_x[i][j]*np.cos(theta) - map_y[i][j]*np.sin(theta) + 16
            map_y_copy[i, j] = map_x[i][j]*np.sin(theta) + map_y[i][j]*np.cos(theta) - 7

    dst = cv2.remap(im, map_x_copy, map_y_copy, cv2.INTER_LINEAR)
    return dst


def rotate_template_90(im):
    h, w, _ = im.shape
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for i in range(map_x.shape[0]):
        map_x[i,:] = [x for x in range(map_x.shape[1])]
    for j in range(map_y.shape[1]):
        map_y[:,j] = [y for y in range(map_y.shape[0])]

    map_x_copy = np.copy(map_x)
    map_y_copy = np.copy(map_y)

    theta = np.pi / 2

    for i in range(map_x.shape[0]):
        for j in range(map_x.shape[1]):
            map_x_copy[i, j] = map_x[i][j]*np.cos(theta) - map_y[i][j]*np.sin(theta) + 32
            map_y_copy[i, j] = map_x[i][j]*np.sin(theta) + map_y[i][j]*np.cos(theta)

    dst = cv2.remap(im, map_x_copy, map_y_copy, cv2.INTER_LINEAR)
    return dst


def order_corners(corners, true_corners):

    # history = np.zeros(4)
    #
    # for tc in true_corners:
    #     best = 0
    #     best_distnace = 1000
    #     i = 0
    #     for c in corners:
    #         distance = euclidean_distance(c, tc)
    #         if distance < best_distnace:
    #             best_distnace = distance
    #             best = i
    #         i += 1
    #     history[best] += 1
    #
    # # (x, y) pairs
    # top_left = corners[0]
    # top_right = corners[0]
    # bottom_left = corners[0]
    # bottom_right = corners[0]
    #
    # top_left_d = 10000
    # top_right_d = 10000
    # bottom_left_d = 10000
    # bottom_right_d = 10000

    corner_index_map = np.zeros(4).astype(np.int)
    true_corner_index_map = np.zeros(4).astype(np.int)
    corner_index_map[:] = -1
    true_corner_index_map[:] = -1
    for _ in range(4):

        best_distance = 10000
        best_true_corner = -1
        best_corner = -1
        i = 0
        for corner in corners:
            if corner_index_map[i] > -1:
                i += 1
                continue
            j = 0
            for true_corner in true_corners:
                distance = euclidean_distance(corner, true_corner)
                if distance < best_distance and true_corner_index_map[j] == -1:
                    best_corner = i
                    best_true_corner = j
                    best_distance = distance
                j += 1

            i += 1
        corner_index_map[best_corner] = best_true_corner
        true_corner_index_map[best_true_corner] = 1

    ordered_corners = []
    for i in range(4):
        index = np.where(corner_index_map == i)[0][0]
        ordered_corners.append(corners[index])

    return ordered_corners

    #     current_top_left_d = euclidean_distance(corner, true_top_left)
    #     current_top_right_d = euclidean_distance(corner, true_top_right)
    #     current_bottom_left_d = euclidean_distance(corner, true_bottom_left)
    #     current_bottom_right_d = euclidean_distance(corner, true_bottom_right)
    #
    #     if current_top_left_d < top_left_d:
    #         top_left_d = current_top_left_d
    #         top_left = corner
    #     if current_top_right_d < top_right_d:
    #         top_right_d = current_top_right_d
    #         top_right = corner
    #     if current_bottom_left_d < bottom_left_d:
    #         bottom_left_d = current_bottom_left_d
    #         bottom_left = corner
    #     if current_bottom_right_d < bottom_right_d:
    #         bottom_right_d = current_bottom_right_d
    #         bottom_right = corner
    #
    # if np.max(history) > 1:
    #     d_1 = euclidean_distance(top_left, top_right)
    #     d_2 = euclidean_distance(top_right, bottom_right)
    #     d_3 = euclidean_distance(bottom_right, bottom_left)
    #     d_4 = euclidean_distance(top_left, bottom_left)
    #     non_ordered_str = str(corners[0]) + str(corners[1]) + str(corners[2]) + str(corners[3])
    #     ordered_str = str(top_left) + str(top_right) + str(bottom_left) + str(bottom_right)
    #     distance_str = str(d_1) + " : " + str(d_2) + " : " + str(d_3) + " : " + str(d_4)
    #     raise RuntimeError("Multiple corners in same quadrant " + non_ordered_str + " | " + ordered_str + " | " + distance_str)
    #
    # # print(str(top_right))
    #
    # return [top_left, bottom_left, top_right, bottom_right]


def draw_circles(circles, img_in):
    sample = np.copy(img_in)
    for circle in circles[0]:
        cv2.circle(sample, (circle[0], circle[1]), circle[2], (255, 0, 0), 1)

    cv2.imwrite("out/sample.png", sample)


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    x = float(p0[0]) - float(p1[0])
    y = float(p0[1]) - float(p1[1])
    return np.sqrt(x ** 2 + y ** 2)

    # raise NotImplementedError


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    h = image.shape[0]
    w = image.shape[1]

    corners = [(0, 0), (0, h-1), (w-1, 0), (w-1, h-1)]

    return corners


def group_points(point, points, values=None, threshold=20):
    has_merged = False
    for p in points:
        x = p.get('x')
        y = p.get('y')
        point_distance = euclidean_distance(point, (x, y))
        if point_distance < threshold:
            value = p.get('value')
            count = p.get('count')
            if values is not None:
                value = (value * count + values[point[1], point[0]]) / (count + 1)
            new_x = (x * count + point[0]) / (count + 1)
            new_y = (y * count + point[1]) / (count + 1)
            p['count'] = count + 1
            p['x'] = new_x
            p['y'] = new_y
            p['value'] = value
            has_merged = True

    if not has_merged:
        value = 0.0
        if values is not None:
            value = values[point[1], point[0]]
        points.append({'x': point[0], 'y': point[1], 'value': value, 'count': 1})

    return points


def group_the_points(point, points, values=None, threshold=20):
    has_merged = False
    x_i, y_i, value_i = point
    i = 0
    for p in points:
        x = p.get('x')
        y = p.get('y')
        value = p.get('value')
        point_distance = euclidean_distance((x_i, y_i), (x, y))
        if point_distance < threshold:
            if value_i < value:
                p['x'] = x_i
                p['y'] = y_i
                p['value'] = value_i
            has_merged = True

    if not has_merged:
        # value = 0.0
        # if values is not None:
        #     value = values[point[1], point[0]]
        points.append({'x': x_i, 'y': y_i, 'value': value_i})
        # points.append((point[0], point[1], value))

    return points


# def filter_specs(gray_image):
#     h = gray_image.shape[0]
#     w = gray_image.shape[1]
#     binary_image = np.zeros((h, w)).astype(np.float)
#     binary_image[gray_image > 1] = 1
#
#     filtered_result = cv2.filter2D(binary_image, -1, SPEC_KERNEL)
#     indices = np.where(filtered_result == 1)
#
#     binary_image[indices] = 0
#     gray = binary_image * 255
#     # cv2.imwrite("out/specs.png", gray)
#
#     reverse_binary = np.ones_like(binary_image)
#     reverse_binary[binary_image == 1] = 0
#
#     filtered_result = cv2.filter2D(reverse_binary, -1, SPEC_KERNEL)
#     indices = np.where(filtered_result == 1)
#     binary_image[indices] = 1
#     # cv2.imwrite("out/reverse.png", binary_image*255)
#
#     result = binary_image*255
#
#     return result.astype(np.uint8)


def draw_corners(corners, image):

    for corner in corners:
        x, y = corner
        cv2.circle(image, (x, y), 2, (0, 0, 0), 5)
        pass

    return image


def draw_grouped_corners(corners, image):

    for corner in corners:
        x = int(corner.get('x'))
        y = int(corner.get('y'))
        cv2.circle(image, (x, y), 2, (0, 0, 0), 5)
        pass

    return image


# def get_best_filter_results(image, kernel, check_for_minimum=True):
#     filter_results = cv2.filter2D(image, -1, kernel)
#     filter_results = np.copy(filter_results)
#
#     if check_for_minimum:
#         filter_results[filter_results > 0] = 0
#         filter_results = filter_results * -1
#
#     result_h = int(filter_results.shape[0] / 2)
#     result_w = int(filter_results.shape[1] / 2)
#     top_left_result = filter_results[0:result_h, 0:result_w]
#     top_right_result = filter_results[0:result_h, result_w:]
#     bottom_left_result = filter_results[result_h:, 0:result_w]
#     bottom_right_result = filter_results[result_h:, result_w:]
#
#     corners = []
#
#     left_top_min = np.where(top_left_result == top_left_result.max())
#     right_top_min = np.where(top_right_result == top_right_result.max())
#     left_bottom_min = np.where(bottom_left_result == bottom_left_result.max())
#     right_bottom_min = np.where(bottom_right_result == bottom_right_result.max())
#
#     lt_corner = (left_top_min[1][0], left_top_min[0][0])
#     rt_corner = (right_top_min[1][0] + result_w, right_top_min[0][0])
#     lb_corner = (left_bottom_min[1][0], left_bottom_min[0][0] + result_h)
#     rb_corner = (right_bottom_min[1][0] + result_w, right_bottom_min[0][0] + result_h)
#
#     corners.append((lt_corner, top_left_result.max()))
#     corners.append((rt_corner, top_right_result.max()))
#     corners.append((lb_corner, bottom_left_result.max()))
#     corners.append((rb_corner, bottom_right_result.max()))
#
#     confidence = np.average([
#         top_left_result.max(), top_right_result.max(), bottom_left_result.max(), bottom_right_result.max()
#     ])
#
#     return corners, confidence


def extract_marker_locations(result, combo_result):
    h, w = result.shape
    template_result = result
    markers = []
    total_value = 0.0
    n = 0
    while n < 4:
        value = template_result.min()
        indices = np.where(template_result == value)
        y = indices[0][0]
        x = indices[1][0]

        y_left = np.min([10, y])
        y_right = np.min([10, h-y])
        x_left = np.min([10, x])
        x_right = np.min([10, w-x])
        template_result[y-y_left:y+y_right, x-x_left:x+x_right] = 1.0

        if euclidean_distance((0, 0), (x,y)) < 10 or euclidean_distance((0, w), (x,y)) < 10 or euclidean_distance((h, 0), (x,y)) < 10 or euclidean_distance((h, w), (x,y)) < 10:
            continue

        # if combo_result[y, x] > 0:
        #     value -= 0.15
        total_value += value
        markers.append({"x": x, "y": y, "value": result[y, x]})

        cv2.imwrite("out/template_result" + str(i) + ".png", template_result * 255)
        n += 1

    confidence = total_value / 4.0
    return markers, confidence


# def extract_marker_locations_1(result, combo_result):
#     template_result = result - combo_result
#     h, w = result.shape
#     # markers = []
#     for threshold in [0.5, 0.6, 0.7]:
#         indices = np.where(template_result < threshold)
#         if len(indices[0]) >= 20:
#             break
#
#     if len(indices[0]) == 0:
#         raise RuntimeError("Error too high")
#
#     points = []
#     for i in range(len(indices[0])):
#         y = indices[0][i]
#         x = indices[1][i]
#         if x == 400 and y == 40:
#             print()
#         value = template_result[y, x]
#         point = (x, y, value)
#         group_the_points(point, points)
#
#     # sorted_p = markers.sort(key = lambda x: x[2])
#     points.sort(key = lambda x: x.get('value'))
#
#     for i in range(len(points)-1, -1, -1):
#         if points[i-1].get('x') == points[i].get('x') and points[i-1].get('y') == points[i].get('y'):
#             del points[i]
#
#     # check for minimums
#     markers = []
#     window = 12
#     for i in range(4):
#         p = points[i]
#         x = p.get('x')
#         y = p.get('y')
#         if x < window or w-x< window or y < window or h-y < window:
#             continue
#         args = np.unravel_index(np.argmin(result[y-window:y+window, x-window:x+window]), shape=(window*2, window*2))
#         x = x + args[1] - window
#         y = y + args[0] - window
#         markers.append({"x": x, "y": y, "value": result[y, x]-combo_result[y,x]})
#
#     return markers


def find_markers(image, template=None, i=0):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    h, w, _ = image.shape
    if h + w > 1200:
        raise RuntimeError("large pic")
    min_radius = 8
    max_radius = 40
    param_2 = 18
    if h + w > 1100:
        min_radius = 18
        param_2 = 20
        max_radius = 65
    # down = cv2.pyrDown(image)
    # down = cv2.pyrDown(down)
    # cv2.imwrite('out/down.png',down)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray,2,9,0.04)
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    image[dst > 0.2*dst.max()]=[0,0,255]
    cv2.imwrite('out/corners.png',image)

    edges = cv2.Canny(gray.astype(np.uint8),90,210)
    cv2.imwrite('out/edges.png',edges)
    circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,3,param1=250,param2=param_2,minRadius=min_radius,maxRadius=max_radius)
    sample = np.zeros_like(image)
    draw_circles(circles, sample)

    circle_edge_combos = np.zeros_like(edges).astype(np.float)
    # for circle in circles[0]:
    #     x = int(circle[0])
    #     y = int(circle[1])
    #     if x < 10 or y < 10 or h-y < 10 or w- x < 10:
    #         continue
    #     if dst[y-5:y+5, x-5:x+5].max() > 0.05*dst.max():
    #         circle_edge_combos[y-2:y+2, x-2:x+2] = 255

    cv2.imwrite('out/combos.png', circle_edge_combos)
    combo_blur = cv2.filter2D(circle_edge_combos, -1, np.ones((5,5)).astype(np.float)/25)
    cv2.imwrite('out/blur.png', combo_blur)

    gray = cv2.copyMakeBorder(gray, 16, 15, 16, 15, cv2.BORDER_CONSTANT)
    gray[0:16, :] = 128

    # combo_blur = combo_blur / (combo_blur.max()*4)
    # gray[gray > 110] = 255
    # gray[gray <= 110] = 0
    # gray[gray > 140] = 255
    # gray[gray <= 140] = 0
    # Index += 1
    # cv2.imwrite('out/gray.png',gray)

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 182, 134   (83, 59)
    t = np.copy(template)

    template = cv2.cvtColor(template.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    template_result = cv2.matchTemplate(gray,template,cv2.TM_SQDIFF_NORMED)
    combo_blur = np.zeros_like(template_result).astype(np.float)
    markers_1, c_1 = extract_marker_locations(template_result, combo_blur)
    cv2.imwrite('out/template_result.png',template_result*255)

    threshold = np.zeros_like(template_result)
    threshold[template_result < np.average(template_result) - 1.5 * np.std(template_result)] = 255
    cv2.imwrite('out/threshold.png',threshold)

    template = rotate_template_90(t)
    template = cv2.cvtColor(template.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    template_result = cv2.matchTemplate(gray,template,cv2.TM_SQDIFF_NORMED)
    markers_2, c_2 = extract_marker_locations(template_result, combo_blur)
    cv2.imwrite('out/template_result_2.png',template_result*255)

    # template = cv2.cvtColor(s.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    # template_result = cv2.matchTemplate(gray,template,cv2.TM_SQDIFF_NORMED)
    # markers_3 = extract_marker_locations(template_result)
    # cv2.imwrite('out/template_result_3.png',template_result*255)
    #
    # template = cv2.cvtColor(t.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    # template_result = cv2.matchTemplate(gray,template,cv2.TM_SQDIFF_NORMED)
    # markers_4 = extract_marker_locations(template_result)
    # cv2.imwrite('out/template_result_4.png',template_result*255)

    if c_1 < 0.3 and c_2 < 0.3:
        raise RuntimeError('low confidence')

    if c_1 < c_2:
        markers = markers_1
    else:
        markers = markers_2
    #
    # b = np.zeros_like(template_result)
    # b[template_result < 0.42] = 255
    # cv2.imwrite('out/b.png', b)
    #
    # gray = cv2.cvtColor(down,cv2.COLOR_BGR2GRAY)
    # template_result_2 = cv2.matchTemplate(gray,template,cv2.TM_SQDIFF_NORMED)
    #
    # down = cv2.pyrDown(down)
    # gray = cv2.cvtColor(down,cv2.COLOR_BGR2GRAY)
    # template_result_3 = cv2.matchTemplate(gray,template,cv2.TM_SQDIFF_NORMED)
    #
    # cv2.imwrite('out/template_result_3.png',template_result_3*255)

    # binary_image = np.zeros_like(gray).astype(np.float)
    # binary_image[gray > 128] = 1
    #
    # corners_min, min_confidence = get_best_filter_results(binary_image, CORNER_KERNEL)
    # corners_max, max_confidence = get_best_filter_results(binary_image, CORNER_KERNEL, False)
    # corners_min_2, min_confidence_2 = get_best_filter_results(binary_image, DIAGONAL_CORNER_KERNEL)
    # corners_max_2, max_confidence_2 = get_best_filter_results(binary_image, DIAGONAL_CORNER_KERNEL, False)
    # # todo add get_best_results for rotation by intermediate amounts?
    #
    # arg_min = np.argmin([min_confidence, max_confidence, min_confidence_2, max_confidence_2])
    # best_corners = [corners_min, corners_max, corners_min_2, corners_max_2][arg_min]

    corners = []

    for m in markers:
        x = m.get('x')
        y = m.get('y')
        corners.append((x,y))

    image = draw_corners(corners, image)
    cv2.imwrite("out/greater" + str(i) + ".png", image)

    true_corners = [(0,0), (0, h), (w, 0), (w, h)]
    ordered_corners = order_corners(corners, true_corners)

    return ordered_corners

    # indices = np.where(abs(result) > 150)


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """

    image_copy = np.copy(image)

    top_left = markers[0]
    bottom_left = markers[1]
    top_right = markers[2]
    bottom_right = markers[3]

    cv2.line(image_copy, top_left, top_right, [0, 0, 255], thickness)
    cv2.line(image_copy, top_left, bottom_left, [0, 0, 255], thickness)
    cv2.line(image_copy, bottom_left, bottom_right, [0, 0, 255], thickness)
    cv2.line(image_copy, top_right, bottom_right, [0, 0, 255], thickness)

    return image_copy


def get_index_matrix(img):
    h = img.shape[0]
    w = img.shape[1]
    index_matrix = np.array(np.where(np.ones((h, w)))).T
    # need to swap x and y indices
    index_x = np.expand_dims(index_matrix[:, 1], axis=1)
    index_y = np.expand_dims(index_matrix[:, 0], axis=1)
    ones = np.expand_dims(np.ones(len(index_matrix)), axis=1)

    final_matrix = np.hstack((index_x, index_y))
    final_matrix = np.hstack((final_matrix, ones))
    return final_matrix


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """

    b_copy = np.copy(imageB)

    index_matrix = get_index_matrix(imageA).astype(np.int)
    transformed_index_matrix = np.matmul(index_matrix, homography.T)

    x_indices = np.round(transformed_index_matrix[:, 0] / transformed_index_matrix[:, 2])
    y_indices = np.round(transformed_index_matrix[:, 1] / transformed_index_matrix[:, 2])

    b_copy[(y_indices.astype(np.int), x_indices.astype(np.int))] = imageA[(index_matrix[:,1], index_matrix[:,0])]

    return b_copy


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """

    mat_A = np.array([
        [0, 0, 0, -src_points[0][0], -src_points[0][1], -1, src_points[0][0]*dst_points[0][1], src_points[0][1]*dst_points[0][1], dst_points[0][1]],
        [-src_points[0][0], -src_points[0][1], -1, 0, 0, 0, src_points[0][0]*dst_points[0][0], src_points[0][1]*dst_points[0][0], dst_points[0][0]],

        [0, 0, 0, -src_points[1][0], -src_points[1][1], -1, src_points[1][0]*dst_points[1][1], src_points[1][1]*dst_points[1][1], dst_points[1][1]],
        [-src_points[1][0], -src_points[1][1], -1, 0, 0, 0, src_points[1][0]*dst_points[1][0], src_points[1][1]*dst_points[1][0], dst_points[1][0]],

        [0, 0, 0, -src_points[2][0], -src_points[2][1], -1, src_points[2][0]*dst_points[2][1], src_points[2][1]*dst_points[2][1], dst_points[2][1]],
        [-src_points[2][0], -src_points[2][1], -1, 0, 0, 0, src_points[2][0]*dst_points[2][0], src_points[2][1]*dst_points[2][0], dst_points[2][0]],

        [0, 0, 0, -src_points[3][0], -src_points[3][1], -1, src_points[3][0]*dst_points[3][1], src_points[3][1]*dst_points[3][1], dst_points[3][1]],
        [-src_points[3][0], -src_points[3][1], -1, 0, 0, 0, src_points[3][0]*dst_points[3][0], src_points[3][1]*dst_points[3][0], dst_points[3][0]],

        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]).astype(dtype=np.float64)

    mat_b = np.zeros((mat_A.shape[0], 1), dtype=np.float64)
    mat_b[8] = 1

    mat_A_inverse = np.linalg.pinv(mat_A)
    x = np.dot(mat_A_inverse, mat_b)

    return np.reshape(x, (3, 3))


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None
    # raise NotImplementedError
