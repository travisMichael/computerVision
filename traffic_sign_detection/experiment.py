"""
CS6476: Problem Set 2 Experiment file

This script contains a series of function calls that run your ps2
implementation and output images so you can verify your results.
"""


import cv2

import ps2


def draw_tl_center(image_in, center, state):
    """Marks the center of a traffic light image and adds coordinates
    with the state of the current image

    Use OpenCV drawing functions to place a marker that represents the
    traffic light center. Additionally, place text using OpenCV tools
    that show the numerical and string values of the traffic light
    center and state. Use the following format:

        ((x-coordinate, y-coordinate), 'color')

    See OpenCV's drawing functions:
    http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    Make sure the font size is large enough so that the text in the
    output image is legible.
    Args:
        image_in (numpy.array): input image.
        center (tuple): center numeric values.
        state (str): traffic light state values can be: 'red',
                     'yellow', 'green'.

    Returns:
        numpy.array: output image showing a marker representing the
        traffic light center and text that presents the numerical
        coordinates with the traffic light state.
    """
    h, w, _ = image_in.shape
    place_left = False
    x = int(center[0])
    y = int(center[1])

    if center[1] > w / 2:
        place_left = True

    if place_left:
        cv2.putText(image_in, '((' + str(x) + ', ' + str(y) + '), ' + state + ')', (x - 180, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    else:
        cv2.putText(image_in, '((' + str(x) + ', ' + str(y) + '), ' + state + ')', (x + 60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.circle(image_in, (x, y), 2, (0, 0, 0), 5)
    return image_in


def mark_traffic_signs(image_in, signs_dict):
    """Marks the center of a traffic sign and adds its coordinates.

    This function uses a dictionary that follows the following
    structure:
    {'sign_name_1': (x, y), 'sign_name_2': (x, y), etc.}

    Where 'sign_name' can be: 'stop', 'no_entry', 'yield',
    'construction', 'warning', and 'traffic_light'.

    Use cv2.putText to place the coordinate values in the output
    image.

    Args:
        image_in (numpy.array): the image to draw on.
        signs_dict (dict): dictionary containing the coordinates of
        each sign found in a scene.

    Returns:
        numpy.array: output image showing markers on each traffic
        sign.
    """

    for key in signs_dict:
        result = signs_dict.get(key)
        if key == 'traffic_light':
            center, state = result
            x = int(center[0])
            y = int(center[1])
            cv2.circle(image_in, (x, y), 2, (0, 0, 0), 5)
            cv2.rectangle(image_in,(x - 50, y - 55),(x + 32, y - 60),(0,0,0),30)
            cv2.putText(image_in, '(' + str(x) + ', ' + str(y) + ')', (x - 59, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            cv2.rectangle(image_in,(x - 50, y + 55),(x + 42, y + 72),(0,0,0),20)
            cv2.putText(image_in, key, (x - 48, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            cv2.putText(image_in, state, (x - 20, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        else:
            x = int(result[0])
            y = int(result[1])
            cv2.circle(image_in, (int(x), int(y)), 2, (0, 0, 0), 5)
            cv2.rectangle(image_in,(x - 50, y - 55),(x + 32, y - 60),(0,0,0),30)
            cv2.putText(image_in, '(' + str(x) + ', ' + str(y) + ')', (x - 59, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            cv2.rectangle(image_in,(x - 50, y + 55),(x + 42, y + 60),(0,0,0),20)
            cv2.putText(image_in, key, (x - 48, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    return image_in


def part_1():

    input_images = ['simple_tl', 'scene_tl_1', 'scene_tl_2', 'scene_tl_3']
    output_labels = ['ps2-1-a-1', 'ps2-1-a-2', 'ps2-1-a-3', 'ps2-1-a-4']

    # Define a radii range, you may define a smaller range based on your
    # observations.
    radii_range = range(10, 30, 1)

    for img_in, label in zip(input_images, output_labels):

        tl = cv2.imread("input_images/{}.png".format(img_in))
        coords, state = ps2.traffic_light_detection(tl, radii_range)

        img_out = draw_tl_center(tl, coords, state)
        cv2.imwrite("{}.png".format(label), img_out)


def part_2():

    input_images = ['scene_dne_1', 'scene_stp_1', 'scene_constr_1',
                    'scene_wrng_1', 'scene_yld_1']

    output_labels = ['ps2-2-a-1', 'ps2-2-a-2', 'ps2-2-a-3', 'ps2-2-a-4',
                     'ps2-2-a-5']

    sign_fns = [ps2.do_not_enter_sign_detection, ps2.stop_sign_detection,
                ps2.construction_sign_detection, ps2.warning_sign_detection,
                ps2.yield_sign_detection]

    sign_labels = ['no_entry', 'stop', 'construction', 'warning', 'yield']

    for img_in, label, fn, name in zip(input_images, output_labels, sign_fns,
                                       sign_labels):

        sign_img = cv2.imread("input_images/{}.png".format(img_in))
        coords = fn(sign_img)

        temp_dict = {name: coords}
        img_out = mark_traffic_signs(sign_img, temp_dict)
        cv2.imwrite("{}.png".format(label), img_out)


def part_3():

    input_images = ['scene_some_signs', 'scene_all_signs']
    output_labels = ['ps2-3-a-1', 'ps2-3-a-2']

    for img_in, label in zip(input_images, output_labels):

        scene = cv2.imread("input_images/{}.png".format(img_in))
        coords = ps2.traffic_sign_detection(scene)

        img_out = mark_traffic_signs(scene, coords)
        cv2.imwrite("{}.png".format(label), img_out)


def part_4():
    input_images = ['scene_some_signs_noisy', 'scene_all_signs_noisy']
    output_labels = ['ps2-4-a-1', 'ps2-4-a-2']

    for img_in, label in zip(input_images, output_labels):
        scene = cv2.imread("input_images/{}.png".format(img_in))
        coords = ps2.traffic_sign_detection_noisy(scene)

        img_out = mark_traffic_signs(scene, coords)
        cv2.imwrite("{}.png".format(label), img_out)


def part_5a():
    input_images = ['img-5-a-1', 'img-5-a-2', 'img-5-a-3']
    output_labels = ['ps2-5-a-1', 'ps2-5-a-2', 'ps2-5-a-3']

    for img_in, label in zip(input_images, output_labels):
        scene = cv2.imread("input_images/{}.png".format(img_in))
        coords = ps2.traffic_sign_detection_challenge(scene)

        img_out = mark_traffic_signs(scene, coords)
        cv2.imwrite("{}.png".format(label), img_out)


def part_5b():
    input_images = ['img-5-b-1', 'img-5-b-2', 'img-5-b-3']
    output_labels = ['ps2-5-b-1', 'ps2-5-b-2', 'ps2-5-b-3']

    for img_in, label in zip(input_images, output_labels):
        scene = cv2.imread("input_images/{}.png".format(img_in))
        coords = ps2.traffic_sign_detection_challenge(scene)

        img_out = mark_traffic_signs(scene, coords)
        cv2.imwrite("{}.png".format(label), img_out)


if __name__ == '__main__':
    # part_1()
    # part_2()
    # part_3()
    # part_4()
    part_5a()
    # part_5b()
