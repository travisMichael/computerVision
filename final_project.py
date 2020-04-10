from removeObject import *

# I = the entire image
# target region = region to be delete (Omega)
# source_region = I - target region (Phi)


# read in an image. it is assumed that the image has marked an object to be removed
# image = get_image()
beach_npy = np.load("One_marker.npy")
h, w, _ = beach_npy.shape
channel_red = beach_npy[:, :, 2]

# Phi = np.zeros((h, w))
# Phi[np.where(channel_red < 230)] = 255
# Phi_c = np.copy(Phi)
# for i in range(10, h-10):
#     for j in range(10, w-10):
#         m = np.median(Phi[i-5:i+5, j-5:j+5])
#         if m > 110:
#             Phi_c[i, j] = 255
# cv2.imwrite("Phi.jpg", Phi_c)
# confidence_matrix = initialize_confidence_matrix(Phi)
#
# image = cv2.imread("One_original_1.jpg").astype(float)
# marker = cv2.imread("One_marker.jpg").astype(float)
#
# image = set_image(marker, image, Phi)
# cv2.imwrite("i.jpg", image)
template_size = 9


def remove_object_from_image_bridge():
    Phi = np.load("bridge/in/bridge_marker.npy")
    original = cv2.imread("bridge/in/bridge.jpg").astype(float)
    marker = cv2.imread("bridge/in/bridge_marker.jpg").astype(float)
    image = set_image(marker, original, Phi)
    confidence_matrix = initialize_confidence_matrix(Phi)
    template_size = 5

    run_algorithm(image, Phi, confidence_matrix, template_size)


def remove_object_from_image_two():
    return None


def remove_object_from_image_three():
    return None


def run_algorithm(image, Phi, confidence_matrix, template_size):
    # confidence_matrix = np.load("confidence_states/c_30.npy")
    Phi = np.load("Phi_states/Phi_228.npy")
    # image = np.load("image_states/beach_patch_filled30.npy").astype(float)
    # cv2.circle(image, (patch_column, patch_row), 2, [0, 0, 255], thickness=2)
    for index in range(0, 230):
        print(index)
        confidence_matrix, image, Phi = patch_iteration(image, Phi, confidence_matrix, template_size)
        np.save("image_states/beach_patch_filled" + str(index), image)
        cv2.imwrite("image_states/beach_patch_filled" + str(index) + ".jpg", image)
        np.save("confidence_states/c_" + str(index), confidence_matrix)
        np.save("Phi_states/Phi_" + str(index), Phi)


if __name__ == "__main__":
    print("hello")

    remove_object_from_image_bridge()



print("done")
