"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np


SPEC_KERNEL = np.array([
    [-1, -1, -1],
    [-1, 1, -1],
    [-1, -1, -1],
]).astype(np.float)

# EMPTY_CIRCLE_KERNEL = np.array([
#     [0, 0, 1, 0, 0],
#     [0, 0, -1, 0, 0],
#     [1, -1, -1, -1, 1],
#     [0, 0, -1, 0, 0],
#     [0, 0, 1, 0, 0],
# ]).astype(np.float)

EMPTY_CIRCLE_KERNEL = np.array([
    [0, 0, 1, 0, 0],
    [0, -1, -1, -1, 1],
    [1, -1, -1, -1, 1],
    [0, -1, -1, -1, 1],
    [0,0, 1, 0, 0],
]).astype(np.float)

DX_KERNEL = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]).astype(np.float)

DY_KERNEL = np.array([
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, -1, 0],
]).astype(np.float)

TEMPLATE_NO_CIRCLE = np.zeros((33, 33, 3))
TEMPLATE_NO_CIRCLE[0:16, 0:16] = [255, 255, 255]
TEMPLATE_NO_CIRCLE[16:,16:] = [255, 255, 255]

TEMPLATE_NO_CIRCLE_90 = np.zeros((33, 33, 3))
TEMPLATE_NO_CIRCLE_90[16:, 0:16] = [255, 255, 255]
TEMPLATE_NO_CIRCLE_90[0:16, 16:] = [255, 255, 255]

s = np.zeros((33, 33, 3))

for i in range(33):
    for j in range(33):
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


def draw_circles(circles, img_in):
    sample = np.copy(img_in)
    for circle in circles[0]:
        cv2.circle(sample, (circle[0], circle[1]), circle[2], 255.0, 1)

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


def draw_corners(corners, image, i):
    im_copy = np.copy(image)

    for corner in corners:
        x, y = corner
        cv2.circle(im_copy, (x, y), 2, (0, 0, 0), 5)
        pass

    cv2.imwrite("out/frame_out/corner" + str(i) + ".png", im_copy)
    return im_copy


def draw_grouped_corners(corners, image):

    for corner in corners:
        x = int(corner.get('x'))
        y = int(corner.get('y'))
        cv2.circle(image, (x, y), 2, (0, 0, 0), 5)
        pass

    return image


def rmse(actual, target):

    diff = actual - target
    squared = diff ** 2
    summed = np.sum(squared, axis=1)
    root = np.sqrt(summed)

    return root


def are_surrounding_pixels_black_and_white(point, image):
    h, w, _ = image.shape
    x, y = point

    sample_size = 20

    y_left = np.min([sample_size, y])
    y_right = np.min([sample_size, h-y])
    x_left = np.min([sample_size, x])
    x_right = np.min([sample_size, w-x])

    left_sample = image[y-y_left:y+y_right, x_left]
    right_sample = image[y-y_left:y+y_right, x_left]
    top_sample = image[y_left, x-x_left:x+x_right]
    bottom_sample = image[y_right, x-x_left:x+x_right]

    black = np.array([0, 0, 0])
    white = np.array([255, 255, 255])

    left_black_error = rmse(left_sample, black)
    left_white_error = rmse(left_sample, white)

    print()

    # each sample must either have black or white pixels


def extract_marker_locations(result, combo_result, image, window=65):
    h, w = result.shape
    template_result = result - combo_result
    markers = []
    total_value = 0.0
    n = 0
    while n < 4:
        value = template_result.min()
        indices = np.where(template_result == value)
        y = indices[0][0]
        x = indices[1][0]

        y_left = np.min([window, y])
        y_right = np.min([window, h-y])
        x_left = np.min([window, x])
        x_right = np.min([window, w-x])
        template_result[y-y_left:y+y_right, x-x_left:x+x_right] = 1.0

        if euclidean_distance((0, 0), (x,y)) < 10 or euclidean_distance((0, w), (x,y)) < 10 or euclidean_distance((h, 0), (x,y)) < 10 or euclidean_distance((h, w), (x,y)) < 10:
            continue

        # if result[y - 15, x] < 0.79 and result[y+15, x] < 0.7:
        #     continue
        # are_surrounding_pixels_black_and_white((x,y), image)

        total_value += value
        markers.append({"x": x, "y": y, "value": result[y, x]})

        cv2.imwrite("out/coverage_result_" + str(n) + ".png", template_result * 255)
        n += 1

    confidence = total_value / 4.0
    return markers, confidence

# 161 128   171, 110

def get_corners_by_threshold(gray, image):
    img_copy = np.copy(image)
    result = cv2.cornerHarris(gray,2,7,0.04)
    result = cv2.dilate(result,None)
    # Threshold for an optimal value, it may vary depending on the image.
    r = np.zeros_like(result)
    r[result > 0.09*result.max()]=255
    cv2.imwrite('out/corners.png',r)
    return r, result


def get_circles_by_threshold(gray, image, threshold, min_r, max_r):
    h = image.shape[0]
    w = image.shape[1]
    new_gray = np.zeros((h + 32, w + 32))
    new_gray[16:h+16, 16:w+16] = gray
    edges = cv2.Canny(new_gray.astype(np.uint8),90,210)
    cv2.imwrite('out/edges.png',edges)
    circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,7,param1=250,param2=16,minRadius=min_r,maxRadius=max_r)
    drawn_circles = np.zeros((h,w)).astype(np.float)
    draw_circles(circles, drawn_circles)

    drawn_circles = cv2.filter2D(drawn_circles, -1, np.ones((5,5)).astype(np.float)/25)
    return circles, drawn_circles


def filter_harris_corners_by_circles(circles, harris_result):
    result = np.copy(harris_result)
    if circles is not None:
        h, w = harris_result.shape
        result = np.zeros_like(harris_result).astype(np.float)
        for circle in circles[0]:
            x = int(circle[0])
            y = int(circle[1])
            if x < 10 or y < 10 or h-y < 10 or w- x < 10:
                continue
            if harris_result[y-5:y+5, x-5:x+5].max() > 0.05*harris_result.max():
                result[y-2:y+2, x-2:x+2] = 255

    cv2.imwrite('out/combos.png', result)
    filtered_result = cv2.filter2D(result, -1, np.ones((5,5)).astype(np.float)/25)
    filtered_result = cv2.filter2D(filtered_result, -1, np.ones((5,5)).astype(np.float)/25)
    cv2.imwrite('out/blur.png', filtered_result)

    filtered_result[filtered_result > 0.01] = 0.2

    return filtered_result / 10


def markers_to_corners_transform(markers):
    corners = []

    for m in markers:
        x = m.get('x')
        y = m.get('y')
        corners.append((x,y))

    return corners


def find_markers(image, template=None, fram_num=0, previous_corners=None):
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
    min_radius = 8
    max_radius = 40
    param_2 = 18
    window = 40
    template_copy = np.copy(template)
    template_list = [template, rotate_template_90(template_copy)]
    if h + w > 1100:
        min_radius = 25
        param_2 = 20
        max_radius = 65
        template_list = [TEMPLATE_NO_CIRCLE, TEMPLATE_NO_CIRCLE_90, s, t]
        window = 255

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    black = np.zeros_like(gray).astype(np.float)
    white = np.zeros_like(gray).astype(np.float)
    black[gray < 110] = 255
    white[gray > 200] = 255
    cv2.imwrite("out/black.png", black)
    cv2.imwrite("out/white.png", white)
    bw = black + white
    cv2.imwrite("out/b_and_w.png", bw)

    b = 3
    bw_with_border = cv2.copyMakeBorder(bw, b, b, b, b, cv2.BORDER_CONSTANT)

    filter_size = 33

    median = cv2.medianBlur(bw_with_border.astype(np.uint8),33)
    cv2.imwrite("out/median.png", median)
    filtered_result = cv2.filter2D(median, -1, np.ones((filter_size,filter_size)).astype(np.float)/(filter_size ** 2))
    filtered_result = filtered_result[b:h+b, b:w+b]
    cv2.imwrite("out/b_and_w_filter.png", filtered_result)
    filtered_result[filtered_result > 90] = 255
    cv2.imwrite("out/b_and_w_filter_thresh.png", filtered_result)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(filtered_result,kernel,iterations = 1)
    erosion[erosion < 240] = 0
    cv2.imwrite("out/filtered_erode.png", erosion)

    filter_size = 15
    filtered_result = cv2.filter2D(erosion, -1, np.ones((filter_size,filter_size)).astype(np.float)/(filter_size ** 2))
    filtered_result[filtered_result < 140] = 0
    cv2.imwrite("out/filtered_erode_2.png", filtered_result)

    threshold = np.zeros_like(filtered_result)
    threshold[filtered_result > 160] = 255
    cv2.imwrite("out/b_and_w_threshold.png", threshold)

    harris_result, raw_harris = get_corners_by_threshold(gray, image)

    harris_copy = np.copy(harris_result).astype(np.float)
    harris_copy[threshold == 0] = 0
    # harris_copy[dI_2_average == 0] = 0
    cv2.imwrite("out/harris_threshold.png", harris_copy)

    filter_size = 51
    threshold = cv2.filter2D(threshold, -1, np.ones((filter_size,filter_size)).astype(np.float)/(filter_size ** 2))
    cv2.imwrite("out/filter_threshold.png", threshold)

    harris_copy[harris_copy > 0] = 0.8
    filter_size = 11
    harris_copy = cv2.filter2D(harris_copy, -1, np.ones((filter_size,filter_size)).astype(np.float)/(filter_size ** 2))
    if harris_copy.max() > 0.5:
        harris_copy = harris_copy / (harris_copy.max())

    # raw_harris = raw_harris / raw_harris.max()
    # raw_harris[raw_harris < 1] = 0.0
    # cv2.imwrite("out/raw.png", raw_harris*255)

    # circles, drawn_circles = get_circles_by_threshold(gray, image, param_2, min_radius, max_radius)

    # BORDER_REFLECT
    # gray_with_border = cv2.copyMakeBorder(gray, 16, 16, 16, 16, cv2.BORDER_CONSTANT)
    # gray_with_border = cv2.copyMakeBorder(gray, 16, 16, 16, 16, cv2.BORDER_REFLECT)
    image_with_border = cv2.copyMakeBorder(np.copy(image), 16, 16, 16, 16, cv2.BORDER_REFLECT)

    # threshold = np.zeros_like(template_result)
    # threshold[template_result < np.average(template_result) - 1.5 * np.std(template_result)] = 255
    # cv2.imwrite('out/threshold.png',threshold)

    template_index = 0
    markers_list = []
    # perform template matching on a variety of different templates
    for template in template_list:
        template_index += 1
        # template = cv2.cvtColor(template.astype(np.uint8),cv2.COLOR_BGR2GRAY)
        template = template.astype(np.uint8)
        template_result = cv2.matchTemplate(image_with_border,template,cv2.TM_SQDIFF_NORMED)
        markers, confidence = extract_marker_locations(template_result, harris_copy, image, window)
        markers_list.append((markers, confidence))
        cv2.imwrite('out/template_result_' + str(template_index) + '.png', template_result*255)

    best_markers = None
    best_confidence = 100
    # now look for best possible set of markers
    for i in range(len(markers_list)):
        markers, confidence = markers_list[i]
        if confidence < best_confidence:
            print(i)
            best_confidence = confidence
            best_markers = markers

    corners = markers_to_corners_transform(best_markers)
    # cv2.imwrite("out/image.png", image)

    draw_corners(corners, image, fram_num)

    true_corners = [(0,0), (0, h), (w, 0), (w, h)]
    if previous_corners is not None:
        true_corners = previous_corners
    ordered_corners = order_corners(corners, true_corners)

    return ordered_corners


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
