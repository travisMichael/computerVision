import numpy as np
import cv2
from scipy import ndimage

r_m = np.array([
    [0, -1],
    [1, 0]
])


def patch_iteration(image, Phi, original_Phi, border, confidence_matrix, template_size):

    Omega = np.zeros_like(Phi)
    Omega[np.where(Phi == 0)] = 255

    # border = initialize_border(Phi)
    border_points = np.where(border == 255)
    d_omega = np.gradient(Omega)
    # border_points = np.where(d_omega[0] + d_omega[1] > 0)

    d_Phi = calculate_3d_gradient(image)

    n_p, d_I_perpindicular = calculate_unit_vectors(border_points, d_omega, d_Phi)

    point_row, point_column = get_highest_patch_priority(border_points, n_p, d_I_perpindicular, confidence_matrix, template_size)

    patch_row, patch_column = find_most_similar_patch(point_row, point_column, image, template_size, Phi, original_Phi)
    border = update_border(point_row, point_column, template_size, border, Phi)

    confidence_matrix, image, Phi = fill_patch(patch_row, patch_column, point_row, point_column, template_size, confidence_matrix, image, Phi)
    border = update_border_with_new_Phi(point_row, point_column, template_size, border, Phi)
    return confidence_matrix, image, Phi, border


def update_border(row, col, template_size, border, Phi):
    original = np.copy(border)
    half = template_size // 2
    # upper and lower
    for i in range(col - half - 1, col + half + 2):
        border[row + half + 1, i] = 255 if Phi[row + half + 1, i] == 0 else 0
        border[row - half - 1, i] = 255 if Phi[row - half - 1, i] == 0 else 0
    # left and right
    for i in range(row - half - 1, row + half + 2):
        border[i, col + half + 1] = 255 if Phi[i, col + half + 1] == 0 else 0
        border[i, col - half - 1] = 255 if Phi[i, col - half - 1] == 0 else 0

    return border


def update_border_with_new_Phi(row, col, template_size, border, Phi):
    original = np.copy(border)
    half = template_size // 2
    border[np.where(Phi == 255)] = 0
    return border


def find_most_similar_patch(point_row, point_col, image, template_size, Phi, original_Phi):
    h = image.shape[0]
    w = image.shape[1]
    half = template_size // 2
    patch = image[point_row-half:point_row+half+1, point_col-half:point_col+half+1, :]
    Phi_patch = Phi[point_row-half:point_row+half+1, point_col-half:point_col+half+1]
    lowest_l2_error = np.inf
    best_row = -1
    best_col = -1

    for i in range(half, h - half):
        for j in range(half, w - half):
            if does_patch_contain_pixels_from_omega(original_Phi[i-half:i+half+1, j-half:j+half+1]):
                continue
            candidate_patch = image[i-half:i+half+1, j-half:j+half+1, :]
            patch_l2_error = calculate_patch_l2_error(patch, candidate_patch, Phi_patch)
            if patch_l2_error < lowest_l2_error:
                lowest_l2_error = patch_l2_error
                best_row = i
                best_col = j

    return best_row, best_col


def does_patch_contain_pixels_from_omega(Phi_patch):
    pixels_in_omega = np.where(Phi_patch == 0)
    if len(pixels_in_omega[0]) > 1:
        return True
    return False


def calculate_patch_l2_error(patch, candidate_patch, Phi_patch):
    h_p = patch.shape[0]
    w_p = patch.shape[1]
    h_cp = candidate_patch.shape[0]
    w_cp = candidate_patch.shape[1]

    if h_p != h_cp or w_p != w_cp:
        return np.inf

    diff_0 = patch[:, :, 0] - candidate_patch[:, :, 0]
    diff_1 = patch[:, :, 1] - candidate_patch[:, :, 1]
    diff_2 = patch[:, :, 2] - candidate_patch[:, :, 2]

    squared_0 = np.multiply(diff_0, diff_0)
    squared_1 = np.multiply(diff_1, diff_1)
    squared_2 = np.multiply(diff_2, diff_2)
    s = squared_0 + squared_1 + squared_2
    s[np.where(Phi_patch == 0)] = 0
    return np.sqrt(np.sum(s))


def fill_patch(patch_row, patch_column, point_row, point_column, template_size, confidence_matrix, image, Phi):
    half = template_size // 2
    Phi_patch = Phi[point_row-half:point_row+half+1, point_column-half:point_column+half+1]
    update_patch = np.copy(image[patch_row-half:patch_row+half+1, patch_column-half:patch_column+half+1])
    update_patch[np.where(Phi_patch == 255)] = 0
    current_patch = image[point_row-half:point_row+half+1, point_column-half:point_column+half+1]
    current_patch[np.where(Phi_patch == 0)] = 0
    image[point_row-half:point_row+half+1, point_column-half:point_column+half+1] = cv2.add(current_patch, update_patch)
    Phi[point_row-half:point_row+half+1, point_column-half:point_column+half+1] = 255
    confidence_matrix = update_confidence_matrix(point_row, point_column, confidence_matrix, template_size)

    return confidence_matrix, image, Phi


def get_highest_patch_priority(border_points, delta_omega_vectors, isophote_vectors, confidence_matrix, template_size):
    number_of_points = len(border_points[0])
    best_point_row = 0
    best_point_column = 0
    max = 0

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
    c_p = np.sum(confidence_patch) / (template_size ** 2)
    return c_p


def calculate_data_term(delta_omega, isophote):
    alpha = 255
    return abs(np.dot(delta_omega, isophote)) / alpha


def initialize_confidence_matrix(Phi):
    h = Phi.shape[0]
    w = Phi.shape[1]
    C = np.zeros((h, w))
    C[np.where(Phi != 0)] = 1
    return C


def update_confidence_matrix(point_row, point_column, confidence_matrix, template_size):
    half = template_size // 2
    confidence_patch = confidence_matrix[point_row-half:point_row+half+1, point_column-half:point_column+half+1]
    c_p = np.sum(confidence_patch) / (template_size * template_size)

    new_confidence_patch = np.zeros_like(confidence_patch)
    new_confidence_patch[np.where(confidence_patch == 0)] = c_p
    confidence_matrix[point_row-half:point_row+half+1, point_column-half:point_column+half+1] += new_confidence_patch
    return confidence_matrix


def calculate_unit_vectors(border_points, d_omega, d_Phi):
    d_omega_orthogonal_unit_vector_list = []
    d_phi_orthogonal_unit_vector_list = []
    number_of_points = len(border_points[0])
    for i in range(number_of_points):
        p_i = border_points[0][i]
        p_j = border_points[1][i]
        # d_omega_vector = # np.array([d_omega[0][p_i, p_j], d_omega[1][p_i, p_j]])
        # three_by_three = border[p_i-1:p_i+2, p_j-1:p_j+2]
        # point_indices = np.where(three_by_three)
        # vector_1 = np.array([point_indices[0][1] - point_indices[0][0], point_indices[1][1] - point_indices[1][0]])
        # vector_2 = np.array([point_indices[0][2] - point_indices[0][1], point_indices[1][2] - point_indices[1][1]])
        # d_omega_vector = vector_1 + vector_2
        d_omega_vector = np.array([d_omega[1][p_i, p_j], d_omega[0][p_i, p_j]])
        d_phi_vector = np.array([d_Phi[1][p_i, p_j], d_Phi[0][p_i, p_j]])

        d_omega_orthogonal_unit_vector_list.append(get_orthogonal_unit_vector(d_omega_vector))
        d_phi_orthogonal_unit_vector_list.append(d_phi_vector)

    return d_omega_orthogonal_unit_vector_list, d_phi_orthogonal_unit_vector_list


def get_orthogonal_unit_vector(vector):
    orthogonal_vector = np.matmul(r_m, vector)

    magnitude = np.linalg.norm(orthogonal_vector)
    if magnitude < 0.0001:
        return orthogonal_vector
    unit_vector = orthogonal_vector / magnitude
    return unit_vector


def calculate_3d_gradient(image):
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

    # dy, dx = np.gradient(r_m)

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


def initialize_border(Phi, remove_isolated_pixels=False):
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

    # sx = ndimage.sobel(Phi, axis=0)
    # sy = ndimage.sobel(Phi, axis=1)
    #
    # border_2 = np.zeros_like(Phi)
    # border_2[np.where(sx != 0)] = 1
    # border_2[np.where(sy != 0)] = 1
    #
    # border_2 = ndimage.binary_erosion(border_2)
    # print()
    # kernel = np.array([
    #     [1, 1, 1],
    #     [1, 0, 1],
    #     [1, 1, 1]
    # ])
    # if remove_isolated_pixels:
    #     while True:
    #         # pixels_removed = 0
    #         condition = np.zeros_like(border, dtype=bool)
    #         output = cv2.filter2D(border, -1, kernel)
    #         condition[np.where(output == 2)] = True
    #         condition[np.where(output == 0)] = True
    #         indices = np.where(condition == False)
    #         if len(indices[0]) < 1:
    #             break
    #         border[indices] = 0
    #
    #     pass

    return border


