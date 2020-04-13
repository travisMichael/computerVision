import numpy as np
import cv2

r_m = np.array([
    [0, -1],
    [1, 0]
])


def patch_iteration(image, Phi, original_Phi, border, confidence_matrix, template_size):

    border_points = np.where(border == 255)

    # Process 4 in pipeline
    n_p, d_I_perpindicular = calculate_point_vectors(border_points, Phi, image)

    # Process 5 in pipeline
    point_row, point_column = get_highest_patch_priority(border_points, n_p, d_I_perpindicular, confidence_matrix, template_size)
    # temporary values used to find the next best patch
    template_height, template_width = get_template_size(point_row, point_column, template_size, Phi)

    # Process 6 in pipeline
    patch = find_most_similar_patch(point_row, point_column, image, template_height, template_width, template_size, Phi, original_Phi)
    # Intermediate calculation to update border
    border = update_border(point_row, point_column, template_height, template_width, border, Phi)

    # Process 7 in pipeline.
    confidence_matrix, image, Phi = fill_patch(patch, point_row, point_column, template_height, template_width, confidence_matrix, image, Phi)
    # Process 8 in pipeline. These updated variables will be passed into the next pipeline iteration
    border[np.where(Phi == 255)] = 0
    return confidence_matrix, image, Phi, border


def get_template_size(point_row, point_column, template_size, Phi):
    h, w = Phi.shape
    half = template_size // 2
    template_height = 0
    template_width = 0
    while template_height < half:
        if point_row - template_height < 0:
            break
        if point_row + template_height >= h-1:
            break
        template_height += 1

    while template_width < half:
        if point_column - template_width < 0:
            break
        if point_column + template_width >= w-1:
            break
        template_width += 1

    return template_height, template_width


def update_border(row, col, template_height, template_width, border, Phi):
    h, w = Phi.shape
    # upper and lower
    for i in range(col - template_width - 1, col + template_width + 2):
        if row + template_height + 1 < h and 0 < i < w:
            border[row + template_height + 1, i] = 255 if Phi[row + template_height + 1, i] == 0 else 0
        if row - template_height - 1 > 0 and 0 < i < w:
            border[row - template_height - 1, i] = 255 if Phi[row - template_height - 1, i] == 0 else 0
    # left and right
    for i in range(row - template_height, row + template_height + 2):
        if row + template_width + 1 < w and 0 < i < h:
            border[i, col + template_width + 1] = 255 if Phi[i, col + template_width + 1] == 0 else 0
        if row - template_width - 1 > 0 and 0 < i < h:
            border[i, col - template_width - 1] = 255 if Phi[i, col - template_width - 1] == 0 else 0

    return border


def find_most_similar_patch(point_row, point_col, image, template_height, template_width, template_size, Phi, original_Phi):
    h = image.shape[0]
    w = image.shape[1]
    res = np.zeros((h, w))
    res[:, :] = np.inf
    in_omega_matrix = np.zeros_like(original_Phi)
    in_omega_matrix[np.where(original_Phi == 0)] = 1
    kernel = np.ones((template_size, template_size))
    output = cv2.filter2D(in_omega_matrix, -1, kernel)

    patch = image[point_row-template_height:point_row+template_height+1, point_col-template_width:point_col+template_width+1, :]
    Phi_patch = Phi[point_row-template_height:point_row+template_height+1, point_col-template_width:point_col+template_width+1]

    res_0 = cv2.matchTemplate(image[:, :, 0].astype("uint8"), patch[:, :, 0].astype("uint8"), method=cv2.TM_SQDIFF, mask=Phi_patch.astype("uint8"))
    res_1 = cv2.matchTemplate(image[:, :, 1].astype("uint8"), patch[:, :, 1].astype("uint8"), method=cv2.TM_SQDIFF, mask=Phi_patch.astype("uint8"))
    res_2 = cv2.matchTemplate(image[:, :, 2].astype("uint8"), patch[:, :, 2].astype("uint8"), method=cv2.TM_SQDIFF, mask=Phi_patch.astype("uint8"))

    res[template_height:h-template_height, template_width:w-template_width] = res_0 + res_1 + res_2
    res[np.where(output > 0.5)] = np.inf

    patch_row, patch_col = np.unravel_index(np.argmin(res, axis=None), res.shape)

    return np.copy(image[patch_row-template_height:patch_row+template_height + 1, patch_col-template_width:patch_col+template_width+1, :])


def fill_patch(patch, point_row, point_column, template_height, template_width, confidence_matrix, image, Phi):
    Phi_patch = Phi[point_row-template_height:point_row+template_height+1, point_column-template_width:point_column+template_width+1]
    patch[np.where(Phi_patch == 255)] = 0
    current_patch = image[point_row-template_height:point_row+template_height+1, point_column-template_width:point_column+template_width+1]
    current_patch[np.where(Phi_patch == 0)] = 0
    image[point_row-template_height:point_row+template_height+1, point_column-template_width:point_column+template_width+1] = cv2.add(current_patch, patch)
    Phi[point_row-template_height:point_row+template_height+1, point_column-template_width:point_column+template_width+1] = 255
    confidence_matrix = update_confidence_matrix(point_row, point_column, confidence_matrix, template_height, template_width)

    return confidence_matrix, image, Phi


def get_highest_patch_priority(border_points, delta_omega_vectors, isophote_vectors, confidence_matrix, template_size):
    number_of_points = len(border_points[0])
    best_point_row = -1
    best_point_column = -1
    max = -1

    for i in range(number_of_points):
        delta_omega = delta_omega_vectors[i]
        isophote = isophote_vectors[i]
        point_row = border_points[0][i]
        point_column = border_points[1][i]
        point_priority = calculate_confidence_term(point_row, point_column, confidence_matrix, template_size) * calculate_data_term(delta_omega, isophote)
        if point_priority > max:
            max = point_priority
            best_point_row = point_row
            best_point_column = point_column

    return best_point_row, best_point_column


# we are assuming that template_size is an odd number
def calculate_confidence_term(point_row, point_column, confidence_matrix, template_size):
    half = template_size // 2
    confidence_patch = confidence_matrix[point_row-half:point_row+half+1, point_column-half:point_column+half+1]
    c_p = np.sum(confidence_patch) / (template_size ** template_size)
    return c_p


def calculate_data_term(orthogonal_delta_omega, delta_phi):
    alpha = 255

    # orthogonal_delta_phi = np.matmul(r_m, delta_phi)
    return abs(np.dot(orthogonal_delta_omega, delta_phi)) / alpha


def initialize_confidence_matrix(Phi):
    h = Phi.shape[0]
    w = Phi.shape[1]
    C = np.zeros((h, w))
    C[np.where(Phi != 0)] = 1
    return C


def update_confidence_matrix(point_row, point_column, confidence_matrix, template_height, template_width):
    confidence_patch = confidence_matrix[point_row-template_height:point_row+template_height+1, point_column-template_width:point_column+template_width+1]
    c_p = np.sum(confidence_patch) / ((template_height*2+1) * (template_width*2+1))

    new_confidence_patch = np.zeros_like(confidence_patch)
    new_confidence_patch[np.where(confidence_patch == 0)] = c_p
    confidence_matrix[point_row-template_height:point_row+template_height+1, point_column-template_width:point_column+template_width+1] += new_confidence_patch
    return confidence_matrix


def calculate_point_vectors(border_points, Phi, image):
    Omega = np.zeros_like(Phi)
    Omega[np.where(Phi == 0)] = 255

    d_omega = np.gradient(Omega)
    isophote = calculate_isophote(image)

    d_omega_unit_vector_list = []
    isophote_vector_list = []
    number_of_points = len(border_points[0])
    for i in range(number_of_points):
        p_i = border_points[0][i]
        p_j = border_points[1][i]
        d_omega_vector = np.array([d_omega[1][p_i, p_j], d_omega[0][p_i, p_j]])
        d_phi_vector = np.array([isophote[1][p_i, p_j], isophote[0][p_i, p_j]])

        d_omega_unit_vector_list.append(get_orthogonal_unit_vector(d_omega_vector))
        isophote_vector_list.append(d_phi_vector)

    return d_omega_unit_vector_list, isophote_vector_list


def get_orthogonal_unit_vector(vector):
    orthogonal_vector = np.matmul(r_m, vector)

    magnitude = np.linalg.norm(orthogonal_vector)
    if magnitude > 0.0001:
        return orthogonal_vector / magnitude
    return orthogonal_vector


def calculate_isophote(image):
    # first, calculate the gradient
    channel_0 = image[:, :, 0]
    channel_1 = image[:, :, 0]
    channel_2 = image[:, :, 0]

    dx_0, dy_0 = np.gradient(channel_0)
    dx_1, dy_1 = np.gradient(channel_1)
    dx_2, dy_2 = np.gradient(channel_2)

    squared_sum = np.multiply(dx_0, dx_0) + np.multiply(dx_1, dx_1) + np.multiply(dx_2, dx_2)
    dx = np.sqrt(squared_sum)

    squared_sum = np.multiply(dy_0, dy_0) + np.multiply(dy_1, dy_1) + np.multiply(dy_2, dy_2)
    dy = np.sqrt(squared_sum)

    # then, shift gradient directions to calculate the isophote
    # (direction perpendicular to gradient directions)
    shift_up = np.zeros_like(dx)
    shift_down = np.zeros_like(dx)
    shift_up[0:3, :] = dx[1:4, :]
    shift_down[1:4, :] = dx[0:3, :]

    dx = dx / 2 + (shift_up + shift_down) / 4

    shift_left = np.zeros_like(dy)
    shift_right= np.zeros_like(dy)
    shift_left[:, 0:3] = dy[:, 1:4]
    shift_right[:, 1:4] = dy[:, 0:3]
    dy = dy / 2 + (shift_up + shift_down) / 4

    return dx, dy


def set_image(marker, original, Phi):
    original[:, :, 0][np.where(Phi == 0)] = marker[:,:,0][np.where(Phi == 0)]
    original[:, :, 1][np.where(Phi == 0)] = marker[:,:,1][np.where(Phi == 0)]
    original[:, :, 2][np.where(Phi == 0)] = marker[:,:,2][np.where(Phi == 0)]
    return original


def initialize_border(Phi):
    h = Phi.shape[0]
    w = Phi.shape[1]

    border = np.zeros_like(Phi)

    for i in range(h):
        hit_edge = False
        for j in range(w):
            if not hit_edge and Phi[i, j] == 0:
                border[i, j] = 255
                hit_edge = True
            elif hit_edge and Phi[i, j] != 0:
                border[i, j - 1] = 255
                hit_edge = False

    for i in range(w):
        hit_edge = False
        for j in range(h):
            if not hit_edge and Phi[j, i] == 0:
                border[j, i] = 255
                hit_edge = True
            elif hit_edge and Phi[j, i] != 0:
                border[j - 1, i] = 255
                hit_edge = False

    return border


