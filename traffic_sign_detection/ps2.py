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

YIELD_LEFT_CORNER_FEATURE_KERNEL = img = np.array([
    [0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
]).astype(np.float)

YIELD_RIGHT_CORNER_FEATURE_KERNEL = img = np.array([
    [0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
]).astype(np.float)

YIELD_BOTTOM_CORNER_FEATURE_KERNEL = img = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]).astype(np.float)

TOP_DIAMOND_CORNER_FEATURE_KERNEL = img = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]).astype(np.float)

RIGHT_DIAMOND_CORNER_FEATURE_KERNEL = img = np.array([
    [0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0],
]).astype(np.float)

BOTTOM_DIAMOND_CORNER_FEATURE_KERNEL = img = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]).astype(np.float)

LEFT_DIAMOND_CORNER_FEATURE_KERNEL = img = np.array([
    [0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0],
]).astype(np.float)

# helper functions ------

def draw_circles(circles, img_in):
    sample = np.copy(img_in)
    for circle in circles[0]:
        cv2.circle(sample, (circle[0], circle[1]), circle[2], (255, 0, 0), 1)

    cv2.imwrite("out/sample.png", sample)


def rmse(actual, target):
    return np.sqrt(((actual - target) ** 2).mean())


def test(edges):
    hough_result = np.copy(edges)
    for i in range(100):
        circles = cv2.HoughCircles(hough_result,cv2.HOUGH_GRADIENT,1,20,param1=250,param2=20,minRadius=10,maxRadius=40)
        if circles is None or len(circles[0]) < 3:
            print("dang")


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


# this function will create a new point or merge points depending on how close the points are
def group_points(point, points):
    has_merged = False
    for p in points:
        x = p.get('x')
        y = p.get('y')
        point_distance = calculate_line_distance(point, (x, y))
        if point_distance < 20:
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

    circles = cv2.HoughCircles(hough_result,cv2.HOUGH_GRADIENT,1,20,param1=250,param2=20,minRadius=5,maxRadius=40)

    if circles is not None:
        for circle in circles[0]:
            circle_list.append(circle)

    return circle_list

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


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    filtered_img = filter_pixels_by_value(img_in, RED).astype(np.uint8)
    cv2.imwrite('out/filtered.png', filtered_img)
    n = cv2.medianBlur(filtered_img, 9)
    # n = cv2.medianBlur(n, 9)
    # cv2.imwrite('out/median.png', n)

    edges = cv2.Canny(n,90,210)
    binary_edges = np.zeros_like(edges)
    binary_edges[edges == 255] = 1
    # should be two results here
    left_corner_feature_result = cv2.filter2D(binary_edges, -1, YIELD_LEFT_CORNER_FEATURE_KERNEL)
    right_corner_feature_result = cv2.filter2D(binary_edges, -1, YIELD_RIGHT_CORNER_FEATURE_KERNEL)
    bottom_corner_feature_result = cv2.filter2D(binary_edges, -1, YIELD_BOTTOM_CORNER_FEATURE_KERNEL)

    left_max = np.max(left_corner_feature_result)
    right_max = np.max(right_corner_feature_result)
    bottom_max = np.max(bottom_corner_feature_result)

    if left_max <= 8:
        raise RuntimeError('Left corner feature not found')

    if right_max <= 8:
        raise RuntimeError('Right corner feature not found')

    if bottom_max <= 8:
        raise RuntimeError('Bottom corner feature not found')

    left_corners = np.where(left_corner_feature_result == left_max)
    right_corners = np.where(right_corner_feature_result == right_max)
    bottom_corners = np.where(bottom_corner_feature_result == bottom_max)

    if len(left_corners[0]) != 2:
        raise RuntimeError('Exactly two left corners were not found')

    if len(right_corners[0]) != 2:
        raise RuntimeError('Exactly two right corners were not found')

    if len(bottom_corners[0]) != 2:
        raise RuntimeError('Exactly two bottom corners were not found')

    if left_corners[0][0] < left_corners[0][1]:
        left_corner = left_corners[1][0], left_corners[0][0]
    else:
        left_corner = left_corners[1][1], left_corners[0][1]

    if right_corners[0][0] < right_corners[0][1]:
        right_corner = right_corners[1][0], right_corners[0][0]
    else:
        right_corner = right_corners[1][1], right_corners[0][1]

    if bottom_corners[0][0] < bottom_corners[0][1]:
        bottom_corner = bottom_corners[1][1], bottom_corners[0][1]
    else:
        bottom_corner = bottom_corners[1][0], bottom_corners[0][0]

    x = ((left_corner[0] + right_corner[0]) / 2 + bottom_corner[0]) / 2
    # y_1 = ((left_corner[1] + right_corner[1]) / 2 + bottom_corner[1]) / 2

    mid_point_top = x, (left_corner[1] + right_corner[1]) / 2
    mid_point_top_to_bottom_line = calculate_cartesian_equation(mid_point_top, bottom_corner)

    left_bottom_mid_point = (left_corner[0] + bottom_corner[0]) / 2, (left_corner[1] + bottom_corner[1]) / 2
    left_bottom_to_right_line = calculate_cartesian_equation(left_bottom_mid_point, right_corner)

    x, y = calculate_intersection_point(mid_point_top_to_bottom_line, left_bottom_to_right_line)

    return x, y


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    filtered_img = filter_pixels_by_value(img_in, RED).astype(np.uint8)
    # cv2.imwrite('out/filtered.png', filtered_img)

    n = cv2.medianBlur(filtered_img, 9)
    n = cv2.medianBlur(n, 9)
    # cv2.imwrite('out/median.png', n)

    edges = cv2.Canny(n,90,210)

    hough_result = np.copy(edges)

    circles = cv2.HoughCircles(hough_result,cv2.HOUGH_GRADIENT,1,20,param1=250,param2=20,minRadius=20,maxRadius=60)

    check_for_circles(circles)
    # draw_circles(circles, img_in)
    # check circles for alternating red-white pixels in circle center
    for circle in circles[0]:
        x = int(circle[0])
        y = int(circle[1])
        r = int(circle[2] / 2)
        temp = img_in[y, x-r:x+r, :]
        if np.average(temp < 170):
            return x, y
        # print(temp)

    return int(circles[0][0][0]), int(circles[0][0][1])


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    filtered_img = filter_pixels_by_value(img_in, YELLOW).astype(np.uint8)
    # cv2.imwrite('out/filtered.png', filtered_img)
    n = cv2.medianBlur(filtered_img, 9)
    # n = cv2.medianBlur(n, 9)
    # cv2.imwrite('out/median.png', n)

    edges = cv2.Canny(n,90,210)
    binary_edges = np.zeros_like(edges)
    binary_edges[edges == 255] = 1

    # TOP_DIAMOND_CORNER_FEATURE_KERNEL
    top_corner_feature_result = cv2.filter2D(binary_edges, -1, TOP_DIAMOND_CORNER_FEATURE_KERNEL)
    right_corner_feature_result = cv2.filter2D(binary_edges, -1, RIGHT_DIAMOND_CORNER_FEATURE_KERNEL)
    bottom_corner_feature_result = cv2.filter2D(binary_edges, -1, BOTTOM_DIAMOND_CORNER_FEATURE_KERNEL)
    left_corner_feature_result = cv2.filter2D(binary_edges, -1, LEFT_DIAMOND_CORNER_FEATURE_KERNEL)

    top_max = np.max(top_corner_feature_result)
    right_max = np.max(right_corner_feature_result)
    bottom_max = np.max(bottom_corner_feature_result)
    left_max = np.max(left_corner_feature_result)

    if top_max < 7:
        raise RuntimeError("top corner less than 7 for construction feature detection")

    if right_max < 7:
        raise RuntimeError("right corner less than 7 for construction feature detection")

    if bottom_max < 7:
        raise RuntimeError("bottom corner less than 7 for construction feature detection")

    if left_max < 7:
        raise RuntimeError("left corner less than 7 for construction feature detection")

    top_corners = np.where(top_corner_feature_result == top_max)
    right_corners = np.where(right_corner_feature_result == right_max)
    bottom_corners = np.where(bottom_corner_feature_result == bottom_max)
    left_corners = np.where(left_corner_feature_result == left_max)

    y = (top_corners[0][0] + right_corners[0][0] + bottom_corners[0][0] + left_corners[0][0]) / 4
    x = (top_corners[1][0] + right_corners[1][0] + bottom_corners[1][0] + left_corners[1][0]) / 4
    return x, y


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    filtered_img = filter_pixels_by_value(img_in, ORANGE).astype(np.uint8)
    # cv2.imwrite('out/filtered.png', filtered_img)
    n = cv2.medianBlur(filtered_img, 9)
    # n = cv2.medianBlur(n, 9)
    # cv2.imwrite('out/median.png', n)

    edges = cv2.Canny(n,90,210)
    binary_edges = np.zeros_like(edges)
    binary_edges[edges == 255] = 1

    # TOP_DIAMOND_CORNER_FEATURE_KERNEL
    top_corner_feature_result = cv2.filter2D(binary_edges, -1, TOP_DIAMOND_CORNER_FEATURE_KERNEL)
    right_corner_feature_result = cv2.filter2D(binary_edges, -1, RIGHT_DIAMOND_CORNER_FEATURE_KERNEL)
    bottom_corner_feature_result = cv2.filter2D(binary_edges, -1, BOTTOM_DIAMOND_CORNER_FEATURE_KERNEL)
    left_corner_feature_result = cv2.filter2D(binary_edges, -1, LEFT_DIAMOND_CORNER_FEATURE_KERNEL)

    top_max = np.max(top_corner_feature_result)
    right_max = np.max(right_corner_feature_result)
    bottom_max = np.max(bottom_corner_feature_result)
    left_max = np.max(left_corner_feature_result)

    if top_max < 7:
        raise RuntimeError("top corner less than 7 for construction feature detection")

    if right_max < 7:
        raise RuntimeError("right corner less than 7 for construction feature detection")

    if bottom_max < 7:
        raise RuntimeError("bottom corner less than 7 for construction feature detection")

    if left_max < 7:
        raise RuntimeError("left corner less than 7 for construction feature detection")

    top_corners = np.where(top_corner_feature_result == top_max)
    right_corners = np.where(right_corner_feature_result == right_max)
    bottom_corners = np.where(bottom_corner_feature_result == bottom_max)
    left_corners = np.where(left_corner_feature_result == left_max)

    y = (top_corners[0][0] + right_corners[0][0] + bottom_corners[0][0] + left_corners[0][0]) / 4
    x = (top_corners[1][0] + right_corners[1][0] + bottom_corners[1][0] + left_corners[1][0]) / 4
    return x, y


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    edges = cv2.Canny(img_in,90,210)

    hough_result = np.copy(edges)

    circles = cv2.HoughCircles(hough_result,cv2.HOUGH_GRADIENT,1,20,param1=250,param2=20,minRadius=10,maxRadius=60)

    check_for_circles(circles)

    # draw_circles(circles, img_in)

    # filter circles
    for circle in circles[0]:
        x = int(circle[0])
        y = int(circle[1])
        radius = circle[2]
        center_value = img_in[y, x, :]
        is_center_white_error = rmse(center_value, WHITE)
        if is_center_white_error > 20:
            continue
        upper_half_value = img_in[y - int(radius / 2), x, :]
        lower_half_value = img_in[y + int(radius / 2), x, :]
        is_upper_half_red_error = rmse(upper_half_value, RED)
        is_lower_half_red_error = rmse(lower_half_value, RED)
        if is_lower_half_red_error < 20 and is_upper_half_red_error < 20:
            return x, y

    raise RuntimeError('No dne signs were found')


def safe_traffic_light_detection(img_in):

    try:
        result = traffic_light_detection(img_in, [10, 30])
        return result
    except RuntimeError:
        print('error in safe traffic light detection')

    return None


def safe_yield_sign_detection(img_in):

    # return yield_sign_detection(img_in)
    try:
        result = yield_sign_detection(img_in)
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
        print('error in yield sign detection')

    return None


def safe_construction_sign_detection(img_in):

    # return construction_sign_detection(img_in)
    try:
        result = construction_sign_detection(img_in)
        return result
    except RuntimeError:
        print('error in yield sign detection')

    return None


def safe_do_not_enter_sign_detection(img_in):

    # return
    try:
        result = do_not_enter_sign_detection(img_in)
        return result
    except RuntimeError:
        print('error in yield sign detection')

    return None


def safe_stop_sign_detection(img_in):

    # return
    try:
        result = stop_sign_detection(img_in)
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
    raise NotImplementedError


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
    raise NotImplementedError
