"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2

# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """

    raise NotImplementedError


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """

    raise NotImplementedError


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    det_threshold = 0.01

    # https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
    k = 3
    grad_x = cv2.Sobel(img_a, cv2.CV_64F, 1, 0, ksize=k, scale=1.0/8.0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img_a, cv2.CV_64F, 0, 1, ksize=k, scale=1.0/8.0, borderType=cv2.BORDER_DEFAULT)
    grad_t = (img_b - img_a) * -1

    # debug_image(grad_x, "grad_x_" + str(k))
    # debug_image(grad_y, "grad_y_" + str(k))

    k = k_size
    grad_y_y = np.multiply(grad_y, grad_y)
    grad_x_x = np.multiply(grad_x, grad_x)
    grad_x_y = np.multiply(grad_x, grad_y)
    # grad_y_y_sum = cv2.filter2D(grad_y_y, -1, np.ones((k, k), dtype=np.float) / (k**2))
    # grad_x_x_sum = cv2.filter2D(grad_x_x, -1, np.ones((k, k), dtype=np.float) / (k**2))
    # grad_x_y_sum = cv2.filter2D(grad_x_y, -1, np.ones((k, k), dtype=np.float) / (k**2))
    grad_y_y_sum = cv2.filter2D(grad_y_y, -1, np.ones((k, k), dtype=np.float))
    grad_x_x_sum = cv2.filter2D(grad_x_x, -1, np.ones((k, k), dtype=np.float))
    grad_x_y_sum = cv2.filter2D(grad_x_y, -1, np.ones((k, k), dtype=np.float))

    m_top = np.stack((grad_x_x_sum, grad_x_y_sum), axis=2)
    m_bottom = np.stack((grad_x_y_sum, grad_y_y_sum), axis=2)

    m = np.stack((m_top, m_bottom), axis=3)

    m_det = np.linalg.det(m)
    m_det = abs(m_det)

    m_inverse = np.zeros_like(m)
    m_inverse[m_det > det_threshold] = np.linalg.inv(m[m_det > det_threshold])

    grad_t_x = np.multiply(grad_x, grad_t)
    grad_t_y = np.multiply(grad_y, grad_t)
    # grad_t_x_sum = cv2.filter2D(grad_t_x, -1, np.ones((k, k), dtype=np.float) / (k**2))
    # grad_t_y_sum = cv2.filter2D(grad_t_y, -1, np.ones((k, k), dtype=np.float) / (k**2))
    grad_t_x_sum = cv2.filter2D(grad_t_x, -1, np.ones((k, k), dtype=np.float))
    grad_t_y_sum = cv2.filter2D(grad_t_y, -1, np.ones((k, k), dtype=np.float))

    b_top = np.expand_dims(grad_t_x_sum, axis=2)
    b_bottom = np.expand_dims(grad_t_y_sum, axis=2)

    b = np.stack((b_top, b_bottom), axis=2)

    x = np.matmul(m_inverse, b)
    u = np.squeeze(x[:, :, 0])
    v = np.squeeze(x[:, :, 1])

    return u, v


def debug_image(image, name):
    cpy = np.copy(image).astype(np.float)
    if cpy.max() > 0.5:
        cpy = cpy / cpy.max() * 255.0

    cv2.imwrite("out/" + name + ".png", cpy)


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    # np.ceil()
    h, w = image.shape

    kernel = np.array([0.05, 0.25, 0.4, 0.25, 0.05])
    kernel = np.outer(kernel, kernel)

    filter_result = cv2.filter2D(image.astype(float), -1, kernel=kernel, borderType=cv2.BORDER_REFLECT101)
    # using numpy index slicing: from:to:step
    down_sampled_image_result = filter_result[0:h+1: 2, 0:w + 1: 2]

    return down_sampled_image_result



def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    pyramid_layers = []

    current_image = np.copy(image).astype(float)
    pyramid_layers.append(current_image)
    # cv2.imwrite("out/pyr_0.png", current_image*255)

    for i in range(levels - 1):
        current_image = reduce_image(current_image)
        pyramid_layers.append(current_image)
        # cv2.imwrite("out/pyr_" + str(i+1) + ".png", current_image*255)

    return pyramid_layers


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    h_final = img_list[0].shape[0]
    w_final = 0
    for image in img_list:
        _, w = image.shape
        w_final += w

    final_image = np.zeros((h_final, w_final), dtype=np.float) + 255
    current_w = 0

    for image in img_list:
        h, w = image.shape
        final_image[0:h, current_w:current_w+w] = normalize_and_scale(image)
        current_w += w

    return final_image


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    h, w = image.shape

    image_new = np.zeros((h * 2, w * 2)).astype(float)
    # for i in range(h):
    #     image_new[i*2][0:w*2 + 1:2] = image[i]
    image_new[0 : h*2 + 1 : 2][0:w*2 + 1:2] = image

    kernel = np.array([0.05, 0.25, 0.4, 0.25, 0.05])
    kernel = np.outer(kernel, kernel)
    blurred = 4 * cv2.filter2D(image_new, -1, kernel=kernel, borderType=cv2.BORDER_REFLECT101)

    return blurred

    # raise NotImplementedError


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """

    raise NotImplementedError


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """

    raise NotImplementedError


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    raise NotImplementedError