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

SPEC_KERNEL = np.array([
    [-1, -1, -1],
    [-1, 1, -1],
    [-1, -1, -1],
]).astype(np.float)


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


def group_points(point, points, threshold=20):
    has_merged = False
    for p in points:
        x = p.get('x')
        y = p.get('y')
        point_distance = euclidean_distance(point, (x, y))
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


def filter_specs(gray_image):
    h = gray_image.shape[0]
    w = gray_image.shape[1]
    binary_image = np.zeros((h, w)).astype(np.float)
    binary_image[gray_image > 1] = 1

    filtered_result = cv2.filter2D(binary_image, -1, SPEC_KERNEL)
    indices = np.where(filtered_result == 1)

    binary_image[indices] = 0
    gray = binary_image * 255
    # cv2.imwrite("out/specs.png", gray)

    reverse_binary = np.ones_like(binary_image)
    reverse_binary[binary_image == 1] = 0

    filtered_result = cv2.filter2D(reverse_binary, -1, SPEC_KERNEL)
    indices = np.where(filtered_result == 1)
    binary_image[indices] = 1
    # cv2.imwrite("out/reverse.png", binary_image*255)

    result = binary_image*255

    return result.astype(np.uint8)


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


def find_markers(image, template=None):
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
    down = cv2.pyrDown(image)
    cv2.imwrite('out/down.png',down)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # gray[gray > 110] = 255
    # gray[gray <= 110] = 0
    # gray[gray > 140] = 255
    # gray[gray <= 140] = 0
    cv2.imwrite('out/gray.png',gray)

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    # template_result = cv2.matchTemplate(image,template,cv2.TM_SQDIFF_NORMED)
    # cv2.imwrite('out/template_result.png',template_result*255)
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

    binary_image = np.zeros_like(gray).astype(np.float)
    binary_image[gray > 128] = 1
    result = cv2.filter2D(binary_image, -1, CORNER_KERNEL)

    result[result > 0] = 0
    result = result * -1

    cv2.imwrite('out/result.png', result)

    indices = np.where(result > 190)

    points = []
    for i in range(len(indices[0])):
        x = indices[1][i]
        y = indices[0][i]
        points.append((x, y))

    grouped = []
    for point in points:
        grouped = group_points(point, grouped)

    corners = []
    for c in grouped:
        x = int(c.get('x'))
        y = int(c.get('y'))
        corners.append((x, y))

    ordered_corners = order_corners(corners, image.shape)

    print()
    return ordered_corners

    # rotated = ndimage.rotate(template, 7, reshape=False)
    # cv2.imwrite('out/roated.png',rotated)
    # template = rotated
    #
    # zeros = np.zeros_like(template_result)
    #
    # zeros[template_result < 0.35] = 255
    # cv2.imwrite('out/matches.png',zeros)
    #
    # # cv2.imwrite('out/template_result.png',res*255)
    # indices = np.where(template_result < 0.1)
    # corners = []
    # for i in range(len(indices[0])):
    #     x = indices[1][i] + 16
    #     y = indices[0][i] + 16
    #     corners.append((x, y))
    #
    # if len(corners) > 4:
    #     raise RuntimeError('More than 4 corner matches were found')
    # if len(corners) < 4:
    #     raise RuntimeError('Less than 4 corner matches were found')
    #
    # return order_corners(corners, image.shape)


def order_corners(corners, img_shape):
    h = img_shape[0]
    w = img_shape[1]
    # (x, y) pairs
    true_top_left = (0, 0)
    true_top_right = (w, 0)
    true_bottom_left = (0, h)
    true_bottom_right = (w, h)

    top_left = corners[0]
    top_right = corners[0]
    bottom_left = corners[0]
    bottom_right = corners[0]

    top_left_d = 10000
    top_right_d = 10000
    bottom_left_d = 10000
    bottom_right_d = 10000

    for corner in corners:
        current_top_left_d = euclidean_distance(corner, true_top_left)
        current_top_right_d = euclidean_distance(corner, true_top_right)
        current_bottom_left_d = euclidean_distance(corner, true_bottom_left)
        current_bottom_right_d = euclidean_distance(corner, true_bottom_right)

        if current_top_left_d < top_left_d:
            top_left_d = current_top_left_d
            top_left = corner
        if current_top_right_d < top_right_d:
            top_right_d = current_top_right_d
            top_right = corner
        if current_bottom_left_d < bottom_left_d:
            bottom_left_d = current_bottom_left_d
            bottom_left = corner
        if current_bottom_right_d < bottom_right_d:
            bottom_right_d = current_bottom_right_d
            bottom_right = corner

    return [top_left, bottom_left, top_right, bottom_right]


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

    raise NotImplementedError


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = None

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    raise NotImplementedError
