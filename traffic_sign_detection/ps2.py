"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np


STOPLIGHT_RED = np.array([0, 0, 255]).astype(np.float)
STOPLIGHT_YELLOW = np.array([0, 255, 255]).astype(np.float)
STOPLIGHT_GREEN = np.array([0, 255, 0]).astype(np.float)
BLACK = np.array([0, 0, 0]).astype(np.float)


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
    # The following sources were used throughout the development of this function
    # https://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
    # https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    # https://docs.opencv.org/master/da/d53/tutorial_py_houghcircles.html

    edges = cv2.Canny(img_in,90,210)

    hough_result = np.copy(edges)

    # test(edges)

    circles = cv2.HoughCircles(hough_result,cv2.HOUGH_GRADIENT,1,20,param1=250,param2=20,minRadius=10,maxRadius=40)

    sample = np.copy(img_in)
    for circle in circles[0]:
        cv2.circle(sample, (circle[0], circle[1]), circle[2], (255, 0, 0), 1)

    cv2.imwrite("out/sample.png", sample)

    stop_light_circles = get_stop_light_circles(circles, img_in)

    red_value = img_in[int(stop_light_circles[0][1]), int(stop_light_circles[0][0]), :]
    yellow_value = img_in[int(stop_light_circles[1][1]), int(stop_light_circles[1][0]), :]
    green_value = img_in[int(stop_light_circles[2][1]), int(stop_light_circles[2][0]), :]

    is_red_error = np.sqrt(((red_value - STOPLIGHT_RED) ** 2).mean())
    is_yellow_error = np.sqrt(((yellow_value - STOPLIGHT_YELLOW) ** 2).mean())
    is_green_error = np.sqrt(((green_value - STOPLIGHT_GREEN) ** 2).mean())

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
    raise NotImplementedError


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    raise NotImplementedError


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    raise NotImplementedError


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    raise NotImplementedError


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
