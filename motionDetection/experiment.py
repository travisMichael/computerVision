"""Problem Set 4: Motion Detection"""

import cv2
import os
import numpy as np

import ps4

# I/O directories
input_dir = "input_images"
output_dir = "./"


# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in range(0, v.shape[0], stride):

        for x in range(0, u.shape[1], stride):

            cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                       y + int(v[y, x] * scale)), color, 1)
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
    return img_out


# Functions you need to complete:

def scale_u_and_v(u, v, level, pyr):
    """Scales up U and V arrays to match the image dimensions assigned 
    to the first pyramid level: pyr[0].

    You will use this method in part 3. In this section you are asked 
    to select a level in the gaussian pyramid which contains images 
    that are smaller than the one located in pyr[0]. This function 
    should take the U and V arrays computed from this lower level and 
    expand them to match a the size of pyr[0].

    This function consists of a sequence of ps4.expand_image operations 
    based on the pyramid level used to obtain both U and V. Multiply 
    the result of expand_image by 2 to scale the vector values. After 
    each expand_image operation you should adjust the resulting arrays 
    to match the current level shape 
    i.e. U.shape == pyr[current_level].shape and 
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Hint: create a for loop from level-1 to 0 inclusive.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:
        u: U array obtained from ps4.optic_flow_lk
        v: V array obtained from ps4.optic_flow_lk
        level: level value used in the gaussian pyramid to obtain U 
               and V (see part_3)
        pyr: gaussian pyramid used to verify the shapes of U and V at 
             each iteration until the level 0 has been met.

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to 
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to 
                             pyr[0].shape
    """

    while level > 0:
        next_image = pyr[level-1]
        u = ps4.expand_image(u) * 2
        v = ps4.expand_image(v) * 2
        u = ps4.truncate_expansion(u, next_image)
        v = ps4.truncate_expansion(v, next_image)
        level -= 1
    return u, v


def part_1a():

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                       'ShiftR2.png'), 0) / 255.
    shift_r5_u5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                          'ShiftR5U5.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    k_size = 55
    k_type = ""
    sigma = 5
    u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)

    q = 0.5
    u = u / q
    v = v / q

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    # input images first.

    k_size = 81
    k_type = ""
    sigma = 0
    u, v = ps4.optic_flow_lk(shift_0, shift_r5_u5, k_size, k_type, sigma)

    q = .5
    u = u / q
    v = v / q

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)


def part_1b():
    """Performs the same operations applied in part_1a using the images
    ShiftR10, ShiftR20 and ShiftR40.

    You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Make sure you explore different parameters and/or pre-process the
    input images to improve your results.

    In this part you should save the following images:
    - ps4-1-b-1.png
    - ps4-1-b-2.png
    - ps4-1-b-3.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    k_size = 171
    k_type = "gaussian"
    sigma = 60
    shift_0 = cv2.GaussianBlur(shift_0,(35,35),10)
    shift_r10 = cv2.GaussianBlur(shift_r10,(35,35),10)
    u, v = ps4.optic_flow_lk(shift_0, shift_r10, k_size, k_type, sigma)

    q = 4.0
    u = u / q
    v = v / q

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-1.png"), u_v)

    k_size = 181
    k_type = "gaussian"
    sigma = 60
    shift_r20 = cv2.GaussianBlur(shift_r20,(35,35),10)
    u, v = ps4.optic_flow_lk(shift_0, shift_r20, k_size, k_type, sigma)

    q = 3.5
    u = u / q
    v = v / q

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-2.png"), u_v)

    k_size = 181
    k_type = "ave"
    sigma = 60
    shift_0 = cv2.GaussianBlur(shift_0,(25,25),15)
    shift_r40 = cv2.GaussianBlur(shift_r40,(25,25),15)
    shift_r40 = cv2.GaussianBlur(shift_r40,(25,25),15)
    u, v = ps4.optic_flow_lk(shift_0, shift_r40, k_size, k_type, sigma)

    q = 2.0
    u = u / q
    v = v / q

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-3.png"), u_v)


def part_2():

    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1',
                                         'yos_img_01.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-a-1.png"),
                yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)

    yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-b-1.png"),
                yos_img_01_l_pyr_img)


def part_3a_1():
    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    levels = 2  # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)

    level_id = 1  # TODO: Select the level number (or id) you wish to use
    k_size = 35 # TODO: Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
                             yos_img_02_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
                ps4.normalize_and_scale(diff_yos_img_01_02))


def part_3a_2():
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    levels = 2  # Define the number of pyramid levels
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
    yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)

    level_id = 1
    k_size = 75
    k_type = ""
    sigma = 0
    u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
                             yos_img_03_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)

    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_4a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    levels = 2  # TODO: Define the number of levels
    k_size = 0  # TODO: Select a kernel size
    k_type = "gaussian"  # TODO: Select a kernel type
    sigma = 30  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    shift_0 = cv2.GaussianBlur(shift_0,(35,35),10)
    shift_r10 = cv2.GaussianBlur(shift_r10,(35,35),10)

    u, v = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    q = 5.0
    u = u / q
    v = v / q
    #
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)
    #
    # # You may want to try different parameters for the remaining function
    # # calls.
    levels = 3
    shift_r20 = cv2.GaussianBlur(shift_r20,(35,35),10)
    u, v = ps4.hierarchical_lk(shift_0, shift_r20, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    q = 10.0
    u = u / q
    v = v / q

    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)

    #

    levels = 3
    shift_r40 = cv2.GaussianBlur(shift_r40,(35,35),10)
    u, v = ps4.hierarchical_lk(shift_0, shift_r40, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    q = 15.0
    u = u / q
    v = v / q
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)


def part_4b():
    urban_img_01 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban01.png'), 0) / 255.
    urban_img_02 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban02.png'), 0) / 255.

    levels = 6
    k_size = 0
    k_type = ""
    sigma = 0
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    # q = 5.0
    # u = u / q
    # v = v / q

    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    # cv2.imwrite(os.path.join(output_dir, "out/warp.png"), ps4.normalize_and_scale(urban_img_02_warped.astype(np.float64)))

    diff_img = urban_img_01 - urban_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
                ps4.normalize_and_scale(diff_img))


def part_5a():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                       'ShiftR10.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    k_size = 91  # TODO: Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    levels = 4
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    # u, v = ps4.optic_flow_lk(shift_0, shift_r10, k_size, k_type, sigma)

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u10, v10, scale=3, stride=10)
    # cv2.imwrite(os.path.join(output_dir, "out/ps4-5-a-1.png"), u_v)

    # cv2.imwrite("out/I_0.png", shift_0*255)

    I_t_02 = interpolate_frames(shift_0, shift_r10, 0.2, u10, v10)
    # cv2.imwrite("out/I_t_02.png", ps4.normalize_and_scale(I_t_02.astype(np.float)))
    I_t_04 = interpolate_frames(shift_0, shift_r10, 0.4, u10, v10)
    # cv2.imwrite("out/I_t_04.png", ps4.normalize_and_scale(I_t_04.astype(np.float)))
    I_t_06 = interpolate_frames(shift_0, shift_r10, 0.6, u10, v10)
    # cv2.imwrite("out/I_t_06.png", I_t_06.astype(np.float) * 255)
    I_t_08 = interpolate_frames(shift_0, shift_r10, 0.8, u10, v10)
    # cv2.imwrite("out/I_t_08.png", I_t_08.astype(np.float) * 255)

    # cv2.imwrite("out/I_1.png", shift_r10*255)

    result = create_2_by_3_image(shift_0, I_t_02, I_t_04, I_t_06, I_t_08, shift_r10)

    cv2.imwrite("ps4-5-a-1.png", result*255)


def create_2_by_3_image(I_1, I_2, I_3, I_4, I_5, I_6):
    h, w = I_1.shape

    image = np.ones((h*2+3, w*3+6))

    image[0:h, 0:w] = I_1
    image[0:h, w+3:w*2+3] = I_2
    image[0:h, w*2+6:w*3+6] = I_3

    image[h+3:h*2+3, 0:w] = I_4
    image[h+3:h*2+3, w+3:w*2+3] = I_5
    image[h+3:h*2+3, w*2+6:w*3+6] = I_6

    return image


def interpolate_frames(I_0, I_1, t, u, v):
    I_1_copy = np.copy(I_1)
    I_0_copy = np.copy(I_0)
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    I_t = ps4.warp(I_0_copy, -t*u, -t*v, interpolation, border_mode)

    return I_t


def part_5b():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    shift_0 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                      'mc01.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                        'mc02.png'), 0) / 255.

    # Optional: smooth the images if LK doesn't work well on raw images
    k_size = 41  # TODO: Select a kernel size
    k_type = ""  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    levels = 4
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    # u, v = ps4.optic_flow_lk(shift_0, shift_r10, k_size, k_type, sigma)

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u10, v10, scale=3, stride=10)
    # cv2.imwrite(os.path.join(output_dir, "out/ps4-5-b-1.png"), u_v)

    # cv2.imwrite("out/I_0.png", shift_0*255)
    I_t_02 = interpolate_frames(shift_0, shift_r10, 0.2, u10, v10)
    # cv2.imwrite("out/I_t_02.png", ps4.normalize_and_scale(I_t_02.astype(np.float)))
    I_t_04 = interpolate_frames(shift_0, shift_r10, 0.4, u10, v10)
    # cv2.imwrite("out/I_t_04.png", ps4.normalize_and_scale(I_t_04.astype(np.float)))
    I_t_06 = interpolate_frames(shift_0, shift_r10, 0.6, u10, v10)
    # cv2.imwrite("out/I_t_06.png", I_t_06.astype(np.float) * 255)
    I_t_08 = interpolate_frames(shift_0, shift_r10, 0.95, u10, v10)
    # cv2.imwrite("out/I_t_08.png", I_t_08.astype(np.float) * 255)
    # cv2.imwrite("out/I_1.png", shift_r10*255)

    result = create_2_by_3_image(shift_0, I_t_02, I_t_04, I_t_06, I_t_08, shift_r10)

    cv2.imwrite("ps4-5-b-1.png", result*255)
    # ----------------------------------------------------------------------------
    shift_0 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                      'mc02.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                        'mc03.png'), 0) / 255.

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u10, v10, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "out/ps4-5-b-2.png"), u_v)

    # cv2.imwrite("out/I_0.png", shift_0*255)
    I_t_02 = interpolate_frames(shift_0, shift_r10, 0.2, u10, v10)
    # cv2.imwrite("out/I_t_02.png", ps4.normalize_and_scale(I_t_02.astype(np.float)))
    I_t_04 = interpolate_frames(shift_0, shift_r10, 0.4, u10, v10)
    # cv2.imwrite("out/I_t_04.png", ps4.normalize_and_scale(I_t_04.astype(np.float)))
    I_t_06 = interpolate_frames(shift_0, shift_r10, 0.6, u10, v10)
    # cv2.imwrite("out/I_t_06.png", I_t_06.astype(np.float) * 255)
    I_t_08 = interpolate_frames(shift_0, shift_r10, 0.95, u10, v10)
    # cv2.imwrite("out/I_t_08.png", I_t_08.astype(np.float) * 255)
    # cv2.imwrite("out/I_1.png", shift_r10*255)

    result = create_2_by_3_image(shift_0, I_t_02, I_t_04, I_t_06, I_t_08, shift_r10)

    cv2.imwrite("ps4-5-b-2.png", result*255)


# Most of this function was taken from the ps3 experiment.py that was given by the OMSCS program at Georgia Tech
def part_6():
    """Challenge Problem

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    video = os.path.join("input_videos", "ps4-my-video.mp4")
    image_gen = ps4.video_frame_generator(video)

    current = image_gen.__next__()
    current_resized = cv2.resize(current, (380, 760))
    next = image_gen.__next__()

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_out = cv2.VideoWriter("output.mp4", fourcc, 20, (380, 760))

    frame_num = 1

    while next is not None:
        # print("Processing fame {}".format(frame_num))

        next_resized = cv2.resize(next, (380, 760))

        k_size = 25
        k_type = ""

        gray_current = cv2.cvtColor(current_resized, cv2.COLOR_BGR2GRAY) / 255
        gray_next = cv2.cvtColor(next_resized, cv2.COLOR_BGR2GRAY) / 255

        if frame_num > 20:
            cv2.imwrite("out/current.png", current_resized)
            cv2.imwrite("out/next.png", next_resized)

        gray_current = cv2.GaussianBlur(gray_current,(35,35),10)
        gray_next = cv2.GaussianBlur(gray_next,(35,35),10)
        u, v = ps4.optic_flow_lk(gray_current, gray_next, k_size, k_type, 0)

        u_abs = abs(u)
        v_abs = abs(v)

        # m = np.max([u_abs.max(), v_abs.max()])
        m = np.max([np.average(u_abs), np.average(v_abs)])
        print("Processing fame {}".format(frame_num), np.average(u), u.max(), np.average(v), v.max())

        u = u / (m)
        v = v / (m)

        u_v = quiver(u, v, scale=3, stride=10)
        cv2.imwrite("out/video/quiver_" + str(frame_num) + ".png", u_v)

        red = current_resized[:,:,2]
        red[u_v[:, :, 1] > 254] = 255
        current_resized[:,:,2] = red
        cv2.imwrite("out/frame/frame" + str(frame_num) + ".png", current_resized)
        video_out.write(current_resized)

        current_resized = next_resized
        next = image_gen.__next__()
        if next is not None:
            next_resized = cv2.resize(next, (380, 760))

        frame_num += 1

    video_out.release()


if __name__ == '__main__':
    part_1a()
    part_1b()
    part_2()
    # part_3a_1()
    # part_3a_2()
    # part_4a()
    # part_4b()
    # part_5a()
    # part_5b()
    # part_6()
