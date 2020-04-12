from removeObject import *


def remove_object_from_image_bridge():
    Phi = np.load("bridge/in/Sarah_marker.npy")
    original = cv2.imread("bridge/in/sarah_r.jpg").astype(float)
    marker = cv2.imread("bridge/in/Sarah_marker2.jpg").astype(float)
    image = set_image(marker, original, Phi)
    confidence_matrix = initialize_confidence_matrix(Phi)
    border = initialize_border(Phi, remove_isolated_pixels=True)
    template_size = 11

    run_algorithm(image, Phi, border, confidence_matrix, template_size, "bridge/out/sarah.jpg")


def remove_object_from_image_anne():
    Phi = np.load("anne/in/anne_marker.npy")
    original = cv2.imread("anne/in/anne_r.jpg").astype(float)
    marker = cv2.imread("anne/in/anne_marker2.jpg").astype(float)
    image = set_image(marker, original, Phi)
    confidence_matrix = initialize_confidence_matrix(Phi)
    border = initialize_border(Phi, remove_isolated_pixels=True)
    template_size = 21

    run_algorithm(image, Phi, border, confidence_matrix, template_size, "anne/out/anne.jpg")


def remove_object_from_image_island():
    Phi = np.load("island/in/island_marker.npy")
    original = cv2.imread("island/in/island_r.jpg").astype(float)
    marker = cv2.imread("island/in/island_marker2.jpg").astype(float)
    image = set_image(marker, original, Phi)
    confidence_matrix = initialize_confidence_matrix(Phi)
    border = initialize_border(Phi, remove_isolated_pixels=True)
    template_size = 9

    run_algorithm(image, Phi, border, confidence_matrix, template_size, "island/out/island.jpg")


def run_algorithm(image, Phi, border, confidence_matrix, template_size, out_file):
    original_Phi = np.copy(Phi)
    np.save("original_phi", Phi)
    # confidence_matrix = np.load("confidence_states/c_229.npy")
    # Phi = np.load("Phi_states/Phi_229.npy")
    # original_Phi = np.load("Phi_states/Phi_229.npy")
    # border = np.load("border.npy")
    # image = np.load("image_states/beach_patch_filled229.npy").astype(float)
    # cv2.circle(image, (patch_column, patch_row), 2, [0, 0, 255], thickness=2)
    for index in range(0, 5000):
        confidence_matrix, image, Phi, border = patch_iteration(image, Phi, original_Phi, border, confidence_matrix, template_size)
        if index % 10 == 0:
            print(index)
            # np.save("border", border)
            # np.save("image_states/beach_patch_filled" + str(index), image)
            cv2.imwrite("image_states/patch_filled" + str(index) + ".jpg", image)
            # np.save("confidence_states/c_" + str(index), confidence_matrix)
            # np.save("Phi_states/Phi_" + str(index), Phi)
        border_points = np.where(border == 255)
        if len(border_points[0]) < 1:
            break

    cv2.imwrite(out_file, image)
    print("done")


if __name__ == "__main__":
    print("hello")

    remove_object_from_image_bridge()



print("done")
