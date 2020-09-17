"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np

# The following sources were used throughout the development of this function
# https://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
# https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
# https://docs.opencv.org/master/da/d53/tutorial_py_houghcircles.html
RED = np.array([0, 0, 255]).astype(np.float)
YELLOW = np.array([0, 255, 255]).astype(np.float)
GREEN = np.array([0, 255, 0]).astype(np.float)
BLACK = np.array([0, 0, 0]).astype(np.float)
WHITE = np.array([255, 255, 255]).astype(np.float)
ORANGE = np.array([0, 128, 255]).astype(np.float)


HAS_TOP_KERNEL = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]).astype(np.float)


HAS_BOTTOM_KERNEL = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
]).astype(np.float)

HAS_LEFT_KERNEL = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]).astype(np.float)

HAS_RIGHT_KERNEL = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]).astype(np.float)


# helper functions ------
def fill_pixels_if_enclosed(binary_image, fill_left_to_right=True, fill_top_to_bottom=True):
    h, w = binary_image.shape

    img_copy = np.copy(binary_image)
    h_zeros = np.zeros((h, 10))
    img_copy = np.hstack((img_copy, h_zeros)).astype(np.uint8)

    result = np.copy(img_copy).astype(np.uint8)

    if fill_top_to_bottom:
        has_top_feature = cv2.filter2D(img_copy, -1, HAS_TOP_KERNEL)
        has_bottom_feature = cv2.filter2D(img_copy, -1, HAS_BOTTOM_KERNEL)
        has_top_and_bottom = np.logical_and(has_top_feature, has_bottom_feature)
        result = np.logical_or(result, has_top_and_bottom).astype(np.uint8) * 255

    if fill_left_to_right:
        has_left_feature = cv2.filter2D(img_copy, -1, HAS_LEFT_KERNEL)
        has_right_feature = cv2.filter2D(img_copy, -1, HAS_RIGHT_KERNEL)
        has_left_and_right = np.logical_and(has_left_feature, has_right_feature)
        result = np.logical_or(result, has_left_and_right).astype(np.uint8) * 255
    return result[:, 0:-10]


def draw_circles(circles, img_in):
    sample = np.copy(img_in)
    for circle in circles[0]:
        cv2.circle(sample, (circle[0], circle[1]), circle[2], (255, 0, 0), 1)

    cv2.imwrite("out/sample.png", sample)


def rmse(actual, target):
    return np.sqrt(((actual - target) ** 2).mean())


# find the return the circles inside the stoplight. circles should be in order, red, yellow, green
def get_stop_light_circles(circles, img_in):

    for i in range(len(circles)):
        if i == 0:
            groups = [{'x': circles[0][0], 'list': [circles[0]]}]
        else:
            insert_new = True
            for j in range(len(groups)):
                group = groups[j]
                x = group.get('x')
                circle_x = circles[i][0]

                if x - 3 <= circle_x <= x + 3:
                    insert_new = False
                    group_list = group.get('list')
                    group_list.append(circles[i])
                    # group['list'] = group_list

            if insert_new:
                groups.append({'x': circles[i][0], 'list': [circles[i]]})

    # make sure there is a group of 3 circles inside a black rectangle
    for group in groups:
        valid_stop_lights = []
        grouped_circles = group.get('list')
        if len(grouped_circles) < 3:
            continue
        for circle in grouped_circles:
            if is_in_black_rectangle(circle, img_in):
                valid_stop_lights.append(circle)

        if len(valid_stop_lights) == 3:
            return sort_stop_lights(valid_stop_lights)

    raise RuntimeError('should have returned valid stop lights')


def is_in_black_rectangle(circle, img):
    x = int(circle[0])
    y = int(circle[1])
    radius = int(circle[2])
    # minus 2 for padding
    img_value = img[y, x - radius - 2, :]

    left_of_radius_is_black_error = rmse(img_value, BLACK)

    # margin of error is high because the stoplight background is not completely black
    if left_of_radius_is_black_error < 65.0:
        return True

    return False


def sort_stop_lights(stop_lights):

    array = np.zeros(3)
    for i in range(len(stop_lights)):
        array[i] = stop_lights[i][1]

    min_arg_value = int(np.argmin(array))
    max_arg_value = int(np.argmax(array))

    s = {0, 1, 2}
    s.remove(min_arg_value)
    s.remove(max_arg_value)
    middle_arg = s.pop()

    return stop_lights[min_arg_value], stop_lights[middle_arg], stop_lights[max_arg_value]


def transform_to_binary_image(img):
    h, w, _ = img.shape
    binary_image = np.zeros((h, w)).astype(np.uint8)

    result = np.sum(img, axis=2)
    binary_image[result > 1] = 1

    return binary_image


def check_for_circles(circles):
    if circles is None:
        raise RuntimeError('No circles were found by hough transform')

    _, number_of_circles, _ = circles.shape

    if number_of_circles < 1:
        raise RuntimeError('No circles were found by hough transform')


def calculate_intersection_point(line_1, line_2):
    if abs(line_1[0] - line_2[0]) < 0.1:
        raise RuntimeError('Cannot find intersection point of parallel lines')

    x = (line_1[1] - line_2[1]) / (line_2[0] - line_1[0])
    y = line_1[0] * x + line_1[1]

    return x, y


def calculate_cartesian_equation(point_1, point_2):
    if point_2[0] - point_1[0] == 0:
        raise RuntimeError('error finding line equation: y = mx + b')
    m = float(point_2[1] - point_1[1]) / float(point_2[0] - point_1[0])
    b = (point_2[1] - (m * point_2[0]))
    return m, b


def calculate_cartesian_equation_new(point_1, point_2):
    if point_2[0] - point_1[0] == 0:
        m = 0
    else:
        m = float(point_2[1] - point_1[1]) / float(point_2[0] - point_1[0])
    b = (point_2[1] - (m * point_2[0]))
    return m, b


# this function will create a new point or merge points depending on how close the points are
def group_points(point, points, threshold=20):
    has_merged = False
    for p in points:
        x = p.get('x')
        y = p.get('y')
        point_distance = calculate_line_distance(point, (x, y))
        if point_distance < threshold:
            count = p.get('count')
            new_x = (x * count + point[0]) / (count + 1)
            new_y = (y * count + point[1]) / (count + 1)
            p['count'] = count + 1
            p['x'] = new_x
            p['y'] = new_y
            has_merged = True

    if not has_merged:
        points.append({'x': point[0], 'y': point[1], 'count': 1})

    return points


def calculate_line_distance(point_1, point_2):
    x = point_1[0] - point_2[0]
    y = point_1[1] - point_2[1]

    return np.sqrt(x**2 + y**2)


# returns circles found in an image or none they don't exist
def get_circles(image, circle_list=None):
    if circle_list is None:
        circle_list = []
    edges = cv2.Canny(image,90,210)
    # cv2.imwrite("out/edges.png", edges)
    hough_result = np.copy(edges)

    circles = cv2.HoughCircles(hough_result,cv2.HOUGH_GRADIENT,1,20,param1=250,param2=20,minRadius=2,maxRadius=35)

    if circles is not None:
        for circle in circles[0]:
            circle_list.append(circle)

    return circle_list


def filter_pixels_by_value(img, value, threshold=55):
    img_copy = np.copy(img).astype(np.float)

    img_with_values = np.zeros_like(img_copy)
    img_with_values[:, :] = value

    diff = img_copy - img_with_values
    squared = diff ** 2
    error = np.sqrt(np.sum(squared, axis=2))

    pixels_to_keep = error < threshold
    new_image = np.zeros_like(img_copy)
    new_image[pixels_to_keep] = value

    return new_image


def merge_colors_in_image(img, color_to_look_for, color_to_change_to, threshold=20):
    img_copy = np.copy(img).astype(np.float)

    img_with_values = np.zeros_like(img).astype(np.float)
    img_with_values[:, :] = color_to_look_for

    diff = img_copy - img_with_values
    squared = diff ** 2
    error = np.sqrt(np.sum(squared, axis=2))

    pixels_to_keep = error < threshold
    # new_image = np.zeros_like(img_copy)
    img_copy[pixels_to_keep] = color_to_change_to

    return img_copy

# End helper functions --------------------------------------------------------------------


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    circle_list = []
    red_img = filter_pixels_by_value(img_in, RED).astype(np.uint8)
    cv2.imwrite("out/filtered.png", red_img)
    yellow_img = filter_pixels_by_value(img_in, YELLOW).astype(np.uint8)
    green_img = filter_pixels_by_value(img_in, GREEN).astype(np.uint8)

    partial_red_img = filter_pixels_by_value(img_in, RED / 2).astype(np.uint8)
    partial_yellow_img = filter_pixels_by_value(img_in, YELLOW / 2).astype(np.uint8)
    partial_green_img = filter_pixels_by_value(img_in, GREEN / 2).astype(np.uint8)

    circle_list = get_circles(red_img, circle_list)
    circle_list = get_circles(yellow_img, circle_list)
    circle_list = get_circles(green_img, circle_list)
    circle_list = get_circles(partial_red_img, circle_list)
    circle_list = get_circles(partial_yellow_img, circle_list)
    circle_list = get_circles(partial_green_img, circle_list)

    if len(circle_list) < 3:
        raise RuntimeError('Expected to detect at least 3 circles for stop light detection')

    stop_light_circles = get_stop_light_circles(circle_list, img_in)

    red_value = img_in[int(stop_light_circles[0][1]), int(stop_light_circles[0][0]), :]
    yellow_value = img_in[int(stop_light_circles[1][1]), int(stop_light_circles[1][0]), :]
    green_value = img_in[int(stop_light_circles[2][1]), int(stop_light_circles[2][0]), :]

    is_red_error = np.sqrt(((red_value - RED) ** 2).mean())
    is_yellow_error = np.sqrt(((yellow_value - YELLOW) ** 2).mean())
    is_green_error = np.sqrt(((green_value - GREEN) ** 2).mean())

    arg_min = np.argmin(np.array([is_red_error, is_yellow_error, is_green_error]))

    state = 'green'

    if arg_min == 0:
        # is red
        state = 'red'
    elif arg_min == 1:
        # is yellow
        state = 'yellow'

    # else is green
    return (stop_light_circles[1][0], stop_light_circles[1][1]), state


# returns the line equations of similar lines
# excludes near vertical lines
def group_lines(line, lines, threshold_m = 0.1, threshold_b = 5):
    x1,y1,x2,y2 = line[0]
    if is_close(x1, x2):
        return lines
    if is_close(y1, y2):
        print("y is close")

    m_primary, b_primary = calculate_cartesian_equation_new((y1, x1), (y2, x2))
    m_primary, b_primary = calculate_cartesian_equation_new((x1, y1), (x2, y2))

    has_merged = False
    for l in lines:
        m = l.get('m')
        b = l.get('b')

        # if is_close(m, m_primary, threshold_m) and is_close(b, b_primary, threshold_b) and False:
        if is_close(m, m_primary, .95):
            count = l.get('count')
            new_m = (m * count + m_primary) / (count + 1)
            new_b = (b * count + b_primary) / (count + 1)
            l['count'] = count + 1
            l['m'] = new_m
            l['b'] = new_b
            has_merged = True

    if not has_merged:
        lines.append({'m': m_primary, 'b': b_primary, 'count': 1})

    return lines


def filter_lines_by_slop(lines, low, high):
    result = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        m, _ = calculate_cartesian_equation_new((x1, y1), (x2, y2))

        if low <= m <= high:
            result.append(line)
    return result


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    red_variant_1 = np.array([70, 68, 160]).astype(np.float)
    red_variant_3 = np.array([40, 26, 134]).astype(np.float)
    red_variant_2 = np.array([59, 71, 175]).astype(np.float)
    red_variant_4 = np.array([70, 75, 230]).astype(np.float)

    merged = merge_colors_in_image(img_in, red_variant_1, RED, threshold=40)
    merged = merge_colors_in_image(merged, red_variant_2, RED, threshold=40)
    merged = merge_colors_in_image(merged, red_variant_3, RED, threshold=40)
    merged = merge_colors_in_image(merged, red_variant_4, RED, threshold=40)
    filtered_img = filter_pixels_by_value(merged, RED, threshold=50).astype(np.uint8)

    binary_filtered_img = transform_to_binary_image(filtered_img)

    enclosed = fill_pixels_if_enclosed(binary_filtered_img)
    enclosed = fill_pixels_if_enclosed(enclosed)
    enclosed = fill_pixels_if_enclosed(enclosed)
    enclosed = fill_pixels_if_enclosed(enclosed)

    edges = cv2.Canny(enclosed,90,210)
    binary_edges = np.zeros_like(edges)
    binary_edges[enclosed == 255] = 1
    # should be two results here

    ln_img = np.zeros_like(img_in)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,20, minLineLength=8, maxLineGap=0)

    if lines is None:
        raise RuntimeError("No lines detected for construction sign detection")

    right_side = filter_lines_by_slop(lines, 1.3, 2.0)
    left_side = filter_lines_by_slop(lines, -2.0, -1.3)
    top_lines = filter_lines_by_slop(lines, -0.01, 0.01)

    grouped_lines = []
    for line in right_side:
        grouped_lines = group_lines(line, grouped_lines)
    for line in left_side:
        grouped_lines = group_lines(line, grouped_lines)
    for line in top_lines:
        grouped_lines = group_lines(line, grouped_lines)
    # plot_lines(grouped_lines)

    right_grouped_lines = []
    for line in right_side:
        right_grouped_lines = group_lines(line, right_grouped_lines)
    left_grouped_lines = []
    for line in left_side:
        left_grouped_lines = group_lines(line, left_grouped_lines)

    if len(right_grouped_lines) != 1:
        raise RuntimeError("right group")
    if len(left_grouped_lines) != 1:
        raise RuntimeError("left group")


    right_line = right_grouped_lines[0]
    left_line = left_grouped_lines[0]

    intersecting_point = calculate_intersection_point((right_line.get('m'),right_line.get('b')), (left_line.get('m'),left_line.get('b')))

    nearest_straight_line = None
    distance = 10000
    for line in top_lines:
        _, y_l, _, _ = line[0]
        if y_l + 35 < intersecting_point[1] and intersecting_point[1] - y_l < distance:
            distance = intersecting_point[1] - y_l
            nearest_straight_line = line

    top_grouped = []
    top_grouped = group_lines(nearest_straight_line, top_grouped)

    grouped_lines = [left_grouped_lines[0], right_grouped_lines[0], top_grouped[0]]

    # grouped_lines = group_lines(nearest_straight_line, [])

    # might have to filter lines before this method
    # hint, we only want lines with a certain slope
    intersecting_points = get_points_of_intersection_from_equations(grouped_lines)

    points = []
    for point in intersecting_points:
        points = group_points((point[0], point[1]), points, threshold=20)

    if len(points) != 3:
        raise RuntimeError("Yield sign should have 3 intersecting points")

    x = 0.0
    y = 0.0
    for point in points:
        x += point.get('x')
        y += point.get('y')

    x = x / len(points)
    y = y / len(points)

    return x, y


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """ # 211, 484 215, 453 np.array([35, 75, 228]).astype(np.float)
    red_variant_1 = np.array([70, 68, 160]).astype(np.float)
    red_variant_3 = np.array([40, 26, 134]).astype(np.float)
    red_variant_2 = np.array([59, 71, 175]).astype(np.float)
    red_variant_4 = np.array([70, 75, 230]).astype(np.float)
    red_variant_5 = np.array([0, 0, 200]).astype(np.float)
    red_variant_6 = np.array([67, 59, 166]).astype(np.float)
    red_variant_7 = np.array([44, 31, 160]).astype(np.float)
    red_variant_8 = np.array([10, 120, 240])

    # merged = merge_colors_in_image(img_in, red_variant_8, BLACK, threshold=15)
    merged = merge_colors_in_image(img_in, red_variant_1, RED, threshold=40)
    merged = merge_colors_in_image(merged, red_variant_2, RED, threshold=40)
    merged = merge_colors_in_image(merged, red_variant_3, RED, threshold=40)
    merged = merge_colors_in_image(merged, red_variant_4, RED, threshold=40)
    merged = merge_colors_in_image(merged, red_variant_5, RED, threshold=40)
    filtered_img = filter_pixels_by_value(merged, RED, threshold=50).astype(np.uint8)

    binary_filtered_img = transform_to_binary_image(filtered_img)

    enclosed = fill_pixels_if_enclosed(binary_filtered_img)
    enclosed = fill_pixels_if_enclosed(enclosed)
    enclosed = fill_pixels_if_enclosed(enclosed)
    enclosed = fill_pixels_if_enclosed(enclosed)

    edges = cv2.Canny(enclosed,90,210)

    hough_result = np.copy(edges)

    circles = cv2.HoughCircles(hough_result,cv2.HOUGH_GRADIENT,1,20,param1=250,param2=19,minRadius=20,maxRadius=90)

    check_for_circles(circles)

    points = []
    for circle in circles[0]:
        points = group_points((circle[0], circle[1]), points, threshold=30)
    draw_circles(circles, img_in)
    # check circles for alternating red-white pixels in circle center
    count = 0
    x_final = 0
    y_final = 0
    for point in points:
        x = int(point.get('x'))
        y = int(point.get('y'))
        # r = int(circle[2] / 2)
        temp = img_in[y, x-8:x+8, :]
        has_white_pixel = False
        has_red_pixel = False

        for i in range(len(temp)):
            is_white_error = rmse(temp[i], WHITE)
            is_red_error = rmse(temp[i], RED)
            if is_white_error < 80:
                has_white_pixel = True
            if is_red_error < 90:
                has_red_pixel = True

        if has_white_pixel and has_red_pixel:
            x_final = (x_final * count + x) / (count + 1)
            y_final = (y_final * count + y) / (count + 1)
            count += 1
    near_orange_error = rmse(img_in[int(y_final), int(x_final), :], red_variant_8)
    near_blue_error = rmse(img_in[int(y_final), int(x_final-5), :], np.array([235, 205, 180]))
    if count >= 1 and near_orange_error > 5 and near_blue_error > 5:
        return x_final, y_final

    raise RuntimeError('Red circles found, but none match stop sign pattern')


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    h, w, _ = img_in.shape
    filtered_img = filter_pixels_by_value(img_in, YELLOW).astype(np.uint8)

    binary_filtered_img = transform_to_binary_image(filtered_img)

    enclosed = fill_pixels_if_enclosed(binary_filtered_img)
    enclosed = fill_pixels_if_enclosed(enclosed)

    binary_filtered_img[enclosed == 255] = 1
    edges = cv2.Canny(enclosed,90,210)
    # binary_edges = np.zeros_like(edges)
    # binary_edges[edges == 255] = 1

    ln_img = np.zeros_like(img_in)

    lines = cv2.HoughLinesP(edges,1,np.pi/180,30)

    if lines is None:
        raise RuntimeError("No lines detected for construction sign detection")

    # might have to filter lines before this method
    # hint, we only want lines with a certain slope
    intersecting_points = get_points_of_intersection(lines)

    points = []
    for point in intersecting_points:
        points = group_points((point[0], point[1]), points, threshold=20)

    if len(points) != 4:
        raise RuntimeError("Warning sign should have 4 intersecting points")

    x = 0.0
    y = 0.0
    for point in points:
        x += point.get('x')
        y += point.get('y')

    x = x / len(points)
    y = y / len(points)

    return x, y


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    orange_variant_1 = np.array([35, 75, 228]).astype(np.float)
    orange_variant_2 = np.array([18, 34, 113]).astype(np.float)

    merged = merge_colors_in_image(img_in, orange_variant_1, ORANGE, threshold=30)
    merged = merge_colors_in_image(merged, orange_variant_2, ORANGE, threshold=30)
    filtered_img = filter_pixels_by_value(merged, ORANGE, threshold=40).astype(np.uint8)

    binary_filtered_img = transform_to_binary_image(filtered_img)
    enclosed = fill_pixels_if_enclosed(binary_filtered_img)
    enclosed = fill_pixels_if_enclosed(enclosed)
    binary_filtered_img[enclosed == 255] = 1

    edges = cv2.Canny(enclosed,90,210)
    binary_edges = np.zeros_like(edges)
    binary_edges[edges == 255] = 1

    ln_img = np.zeros_like(img_in)

    lines = cv2.HoughLinesP(edges,1,np.pi/180,40)

    if lines is None:
        raise RuntimeError("No lines detected for construction sign detection")


    # might have to filter lines before this method
    # hint, we only want lines with a certain slope
    intersecting_points = get_points_of_intersection(lines)

    points = []
    for point in intersecting_points:
        points = group_points((point[0], point[1]), points, threshold=20)

    if len(points) != 4:
        raise RuntimeError("Construction sign should have 4 intersecting points")

    x = 0.0
    y = 0.0
    for point in points:
        x += point.get('x')
        y += point.get('y')

    x = x / len(points)
    y = y / len(points)

    return x, y


def is_close(value_1, value_2, threshold=0.1):
    diff = abs(value_1 - value_2)
    if diff < threshold:
        return True
    return False


def get_points_of_intersection(lines):
    intersecting_points = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        for l in lines:
            x_1,y_1,x_2,y_2 = l[0]
            if x1 == x_1 and y1 == y_2 and x2 == x_2 and y2 == y_2:
                continue
            if is_close(y1, y2) or is_close(y_1, y_2):
                continue
            m1, b1 = calculate_cartesian_equation_new((x1, y1), (x2, y2))
            m2, b2 = calculate_cartesian_equation_new((x_1, y_1), (x_2, y_2))
            if is_close(m1, m2):
                continue
            point = calculate_intersection_point((m1, b1), (m2, b2))
            intersecting_points.append(point)

    return intersecting_points


def get_points_of_intersection_from_equations(line_equations):
    intersecting_points = []
    for e1 in line_equations:
        m1 = e1.get('m')
        b1 = e1.get('b')
        for e2 in line_equations:
            m2 = e2.get('m')
            b2 = e2.get('b')
            if m1 == m2 and b1 == b2:
                continue

            point = calculate_intersection_point((m1, b1), (m2, b2))
            intersecting_points.append(point)

    return intersecting_points


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    red_variant_1 = np.array([70, 68, 160]).astype(np.float)
    red_variant_3 = np.array([40, 26, 134]).astype(np.float)
    red_variant_2 = np.array([59, 71, 175]).astype(np.float)
    red_variant_4 = np.array([70, 75, 230]).astype(np.float)
    red_variant_8 = np.array([10, 120, 240])

    merged = merge_colors_in_image(img_in, red_variant_1, RED, threshold=40)
    merged = merge_colors_in_image(merged, red_variant_2, RED, threshold=40)
    merged = merge_colors_in_image(merged, red_variant_3, RED, threshold=40)
    merged = merge_colors_in_image(merged, red_variant_4, RED, threshold=40)
    filtered_img = filter_pixels_by_value(merged, RED, threshold=50).astype(np.uint8)

    binary_filtered_img = transform_to_binary_image(filtered_img)

    enclosed = fill_pixels_if_enclosed(binary_filtered_img)
    enclosed = fill_pixels_if_enclosed(enclosed)

    edges = cv2.Canny(enclosed,90,210)

    hough_result = np.copy(edges)

    circles = cv2.HoughCircles(hough_result,cv2.HOUGH_GRADIENT,1,20,param1=250,param2=20,minRadius=10,maxRadius=60)

    check_for_circles(circles)

    # filter circles
    for circle in circles[0]:
        x = int(circle[0])
        y = int(circle[1])
        radius = circle[2]

        temp = img_in[y, x-5:x+5, :]
        has_white_pixel = False
        has_red_pixel = False

        for i in range(len(temp)):
            is_white_error = rmse(temp[i], WHITE)
            is_red_error = rmse(temp[i], RED)
            if is_white_error < 80:
                has_white_pixel = True
            if is_red_error < 90:
                has_red_pixel = True

        if has_red_pixel or not has_white_pixel:
            continue

        upper_half_value = img_in[y - int(radius / 2), x, :]
        lower_half_value = img_in[y + int(radius / 2), x, :]
        is_upper_half_red_error = rmse(upper_half_value, RED)
        is_lower_half_red_error = rmse(lower_half_value, RED)
        is_upper_half_orange_error = rmse(upper_half_value, red_variant_8)
        is_lower_half_orange_error = rmse(lower_half_value, red_variant_8)
        # near_orange_error = rmse(img_in[int(x), int(y), :], red_variant_8)
        if is_upper_half_orange_error < 5 or is_lower_half_orange_error < 5:
            continue
        if is_lower_half_red_error < 100 and is_upper_half_red_error < 100:
            return x, y
        if enclosed[y + int(radius / 2), x] == 255 and enclosed[y - int(radius / 2), x] == 255:
            return x, y

    raise RuntimeError('No dne signs were found')


def safe_traffic_light_detection(img_in):

    try:
        # blur = cv2.GaussianBlur(img_in,(5,5),0)
        median = cv2.medianBlur(img_in,5)
        result = traffic_light_detection(median, [10, 30])
        return result
    except RuntimeError:
        print('error in safe traffic light detection')

    return None


def safe_yield_sign_detection(img_in):

    # return yield_sign_detection(img_in)
    try:
        # blur = cv2.GaussianBlur(img_in,(5,5),0)
        median = cv2.medianBlur(img_in,5)
        result = yield_sign_detection(median)
        return result
    except RuntimeError:
        print('error in yield sign detection')

    return None


def safe_warning_sign_detection(img_in):

    # return warning_sign_detection(img_in)
    try:
        result = warning_sign_detection(img_in)
        return result
    except RuntimeError:
        print('error in warning sign detection')

    return None


def safe_construction_sign_detection(img_in):

    # return construction_sign_detection(img_in)
    try:
        blur = cv2.GaussianBlur(img_in,(5,5),0)
        result = construction_sign_detection(blur)
        return result
    except RuntimeError:
        print('error in construction sign detection')

    return None


def safe_do_not_enter_sign_detection(img_in):

    # return
    try:
        result = do_not_enter_sign_detection(img_in)
        return result
    except RuntimeError:
        print('error in dne sign detection')

    return None


def safe_stop_sign_detection(img_in):

    # return
    try:
        median = cv2.medianBlur(img_in,5)
        result = stop_sign_detection(median)
        return result
    except RuntimeError:
        print('error in stop sign detection')

    return None


def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    result = {}
    traffic_light_result = safe_traffic_light_detection(img_in)
    if traffic_light_result is not None:
        # result['traffic_light'] = (traffic_light_result[0][0], traffic_light_result[0][1])
        result['traffic_light'] = traffic_light_result

    construction_sign_result = safe_construction_sign_detection(img_in)
    if construction_sign_result is not None:
        result['construction'] = (construction_sign_result[0], construction_sign_result[1])

    warning_sign_result = safe_warning_sign_detection(img_in)
    if warning_sign_result is not None:
        result['warning'] = (warning_sign_result[0], warning_sign_result[1])

    do_not_enter_sign_result = safe_do_not_enter_sign_detection(img_in)
    if do_not_enter_sign_result is not None:
        result['no_entry'] = (do_not_enter_sign_result[0], do_not_enter_sign_result[1])

    stop_sign_result = safe_stop_sign_detection(img_in)
    if stop_sign_result is not None:
        result['stop'] = (stop_sign_result[0], stop_sign_result[1])

    yield_sign_result = safe_yield_sign_detection(img_in)
    if yield_sign_result is not None:
        result['yield'] = (yield_sign_result[0], yield_sign_result[1])

    return result


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    result = traffic_sign_detection(img_in)
    return result


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """

    result = traffic_sign_detection(img_in)
    return result
