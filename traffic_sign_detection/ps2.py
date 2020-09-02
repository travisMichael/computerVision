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

    edges = cv2.Canny(img_in,90,210)

    hough_result = np.copy(edges)

    # test(edges)

    circles = cv2.HoughCircles(hough_result,cv2.HOUGH_GRADIENT,1,20,param1=250,param2=20,minRadius=10,maxRadius=40)

    # sample = np.copy(img_in)
    # for circle in circles[0]:
    #     cv2.circle(sample, (circle[0], circle[1]), circle[2], (255, 0, 0), 1)
    #
    # cv2.imwrite("out/sample.png", sample)

    stop_light_circles = get_stop_light_circles(circles, img_in)

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
    # filter circles by column alignment
    # for each, circle, move outward until dark pixel is encountered

    # hough function returns 3D array for some reason?
    circles = circles[0]

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

    raise Exception('should have returned valid stop lights')


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


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    raise NotImplementedError


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    filtered_img = filter_pixels_by_value(img_in, RED).astype(np.uint8)
    cv2.imwrite('out/filtered.png', filtered_img)

    n = cv2.medianBlur(filtered_img, 9)
    n = cv2.medianBlur(n, 9)
    cv2.imwrite('out/median.png', n)

    edges = cv2.Canny(n,90,210)

    hough_result = np.copy(edges)

    circles = cv2.HoughCircles(hough_result,cv2.HOUGH_GRADIENT,1,20,param1=250,param2=20,minRadius=20,maxRadius=60)

    check_for_circles(circles)
    # draw_circles(circles, img_in)
    # check circles for alternating red-white pixels in circle center

    return int(circles[0][0][0]), int(circles[0][0][1])
    # raise NotImplementedError


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
    hough_result = np.copy(edges)
    # cv2.imwrite('out/edges.png', edges)

    line_image = np.zeros_like(filtered_img)
    lines = cv2.HoughLinesP(hough_result, 1, np.pi / 45, 50, None, 0, 0)

    # for i in range(0, len(lines)):
    #     l = lines[i][0]
    #     cv2.line(line_image, (l[0], l[1]), (l[2], l[3]), (0,255,255), 3, cv2.LINE_AA)
    #
    #     cv2.imwrite('out/lines' + str(i) + '.png', line_image)

    points = transform_line_to_points(lines)
    parallelogram_points = construct_parallelogram_point_map(points)
    left = parallelogram_points.get('left')
    top = parallelogram_points.get('top')
    right = parallelogram_points.get('right')
    bottom = parallelogram_points.get('bottom')
    x_center = (left.get('x') + top.get('x') + right.get('x') + bottom.get('x')) * 1.0 / 4.0
    y_center = (left.get('y') + top.get('y') + right.get('y') + bottom.get('y')) * 1.0 / 4.0
    return x_center, y_center


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
    hough_result = np.copy(edges)
    # cv2.imwrite('out/edges.png', edges)

    line_image = np.zeros_like(filtered_img)
    lines = cv2.HoughLinesP(hough_result, 1, np.pi / 45, 50, None, 0, 0)

    # for i in range(0, len(lines)):
    #     l = lines[i][0]
    #     cv2.line(line_image, (l[0], l[1]), (l[2], l[3]), (0,255,255), 3, cv2.LINE_AA)
    #
    #     cv2.imwrite('out/lines' + str(i) + '.png', line_image)

    points = transform_line_to_points(lines)
    parallelogram_points = construct_parallelogram_point_map(points)
    left = parallelogram_points.get('left')
    top = parallelogram_points.get('top')
    right = parallelogram_points.get('right')
    bottom = parallelogram_points.get('bottom')
    x_center = (left.get('x') + top.get('x') + right.get('x') + bottom.get('x')) * 1.0 / 4.0
    y_center = (left.get('y') + top.get('y') + right.get('y') + bottom.get('y')) * 1.0 / 4.0
    return x_center, y_center
    # raise NotImplementedError


def construct_parallelogram_point_map(points):
    if len(points) != 4:
        raise RuntimeError('parallelograms require 4 points')

    # calculate left
    point_map = {}
    furthest_left = 10000
    furthest_right = -1
    highest = 10000
    lowest = -1
    furthest_left_point = None
    furthest_right_point = None
    highest_point = None
    lowest_point = None
    for point in points:
        x = point.get('x')
        y = point.get('y')
        if x < furthest_left:
            furthest_left = x
            furthest_left_point = point
        if x > furthest_right:
            furthest_right = x
            furthest_right_point = point
        if y < highest:
            highest = y
            highest_point = point
        if y > lowest:
            lowest = y
            lowest_point = point

    point_map['left'] = furthest_left_point
    point_map['top'] = highest_point
    point_map['right'] = furthest_right_point
    point_map['bottom'] = lowest_point
    return point_map


def transform_line_to_points(lines):
    n = len(lines)
    points = []
    line_equations = []

    for i in range(0, n):
        l = lines[i][0]
        equation = calculate_cartesian_equation((l[0], l[1]), (l[2], l[3]))
        line_equations.append(equation)

    for i in range(0, n):
        equation_1 = line_equations[i]
        for j in range(0, n):
            equation_2 = line_equations[j]
            if abs(equation_1[0] - equation_2[0]) < 0.2:
                continue
            x, y = calculate_intersection_point(equation_1, equation_2)
            points = group_points((x, y), points)

    # for i in range(0, len(lines)):
    #     l = lines[i][0]
    #     line_distance = calculate_line_distance((l[0], l[1]), (l[2], l[3]))
    #     if line_distance < 9:
    #         continue
    #     if len(points) == 0:
    #         points.append({'x': l[0], 'y': l[1], 'count': 1})
    #         points.append({'x': l[2], 'y': l[3], 'count': 1})
    #     else:
    #         # are points close to any other point? if so merge points, else add new point
    #         points = group_points((l[0], l[1]), points)
    #         points = group_points((l[2], l[3]), points)

    return points


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

    # print(circles)
    # raise NotImplementedError


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
    raise NotImplementedError


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
