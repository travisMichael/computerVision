from removeObject import *
from marker import *


def remove_object_from_image_shower():
    original = cv2.imread("shower/in/shower_original.jpg").astype(float)
    marker, Phi = get_algorithm_params_for_shower()
    image = set_image(marker, original, Phi)
    confidence_matrix = initialize_confidence_matrix(Phi)
    border = initialize_border(Phi)
    template_size = 7
    run_algorithm(image, Phi, border, confidence_matrix, template_size, "shower/out/shower_output.jpg")


def remove_object_from_image_anne():
    original = cv2.imread("anne/in/anne_r.jpg").astype(float)
    marker, Phi = get_algorithm_params_for_anne()
    image = set_image(marker, original, Phi)
    confidence_matrix = initialize_confidence_matrix(Phi)
    border = initialize_border(Phi)
    template_size = 21

    run_algorithm(image, Phi, border, confidence_matrix, template_size, "anne/out/anne.jpg")


def remove_object_from_image_island():
    original = cv2.imread("island/in/island_r.jpg").astype(float)
    marker, Phi = get_algorithm_params_for_island()
    image = set_image(marker, original, Phi)
    confidence_matrix = initialize_confidence_matrix(Phi)
    border = initialize_border(Phi)
    template_size = 9

    run_algorithm(image, Phi, border, confidence_matrix, template_size, "island/out/island.jpg")


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
    print("hello")

    remove_object_from_image_shower()

    print("done")
