from object_removal.removeObject import *
from object_removal.marker import *
import sys


def remove_object_from_image_shower():
    original = cv2.imread("image_set_2/shower_original.jpg").astype(float)
    marker, Phi = get_algorithm_params_for_shower()
    image = set_image(marker, original, Phi)
    confidence_matrix = initialize_confidence_matrix(Phi)
    border = initialize_border(Phi)
    template_size = 7
    run_algorithm(image, Phi, border, confidence_matrix, template_size, "image_set_2/shower_output.jpg")


def remove_object_from_image_anne():
    original = cv2.imread("image_set_3/anne_original.jpg").astype(float)
    marker, Phi = get_algorithm_params_for_anne()
    image = set_image(marker, original, Phi)
    confidence_matrix = initialize_confidence_matrix(Phi)
    border = initialize_border(Phi)
    template_size = 11

    run_algorithm(image, Phi, border, confidence_matrix, template_size, "image_set_3/anne_output.jpg")


def remove_object_from_image_island():
    original = cv2.imread("image_set_1/island_original.jpg").astype(float)
    marker, Phi = get_algorithm_params_for_island()
    image = set_image(marker, original, Phi)
    confidence_matrix = initialize_confidence_matrix(Phi)
    border = initialize_border(Phi)
    template_size = 11

    run_algorithm(image, Phi, border, confidence_matrix, template_size, "image_set_1/island_output.jpg")


def run_algorithm(image, Phi, border, confidence_matrix, template_size, out_file):
    # This variable is used to assist in finding a candidate patch, which must be found in the original source pixels.
    original_Phi = np.copy(Phi)

    # Remove patches until there are no pixels left in the border
    while len(np.where(border == 255)[0]) > 1:
        confidence_matrix, image, Phi, border = patch_iteration(image, Phi, original_Phi, border, confidence_matrix, template_size)

    # Save the output of the algorithm
    cv2.imwrite(out_file, image)

# def run_algorithm(image, Phi, border, confidence_matrix, template_size, out_file):
#     original_Phi = np.copy(Phi)
#
#     for index in range(0, 5000):
#         confidence_matrix, image, Phi, border = patch_iteration(image, Phi, original_Phi, border, confidence_matrix, template_size)
#         if index % 10 == 0:
#             print(index)
#             cv2.imwrite("image_states/patch_filled" + str(index) + ".jpg", image)
#
#         border_points = np.where(border == 255)
#         if len(border_points[0]) < 1:
#             break
#
#     cv2.imwrite(out_file, image)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Not enough arguments. Exiting.")

    if sys.argv[1] == "island":
        remove_object_from_image_island()

    if sys.argv[1] == "shower":
        remove_object_from_image_shower()

    if sys.argv[1] == "anne":
        remove_object_from_image_anne()

    print("done")
