"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import cv2
import ps5
import os
import numpy as np
import multi_filter
# import kalman
# import kalman_2

# I/O directories
input_dir = "input_images"
output_dir = "./"

NOISE_1 = {'x': 2.5, 'y': 2.5}
NOISE_2 = {'x': 7.5, 'y': 7.5}


# Helper code
def run_particle_filter(filter_class, imgs_dir, template_rect,
                        save_frames={}, **kwargs):
    """Runs a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any
    keyword arguments.

    Do not modify this function except for the debugging flag.

    Args:
        filter_class (object): particle filter class to instantiate
                           (e.g. ParticleFilter).
        imgs_dir (str): path to input images.
        template_rect (dict): template bounds (x, y, w, h), as float
                              or int.
        save_frames (dict): frames to save
                            {<frame number>|'template': <filename>}.
        **kwargs: arbitrary keyword arguments passed on to particle
                  filter class.

    Returns:
        None.
    """

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    # Initialize objects
    template = None
    pf = None
    frame_num = 0

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Extract template and initialize (one-time only)
        if template is None:
            template = frame[int(template_rect['y']):
                             int(template_rect['y'] + template_rect['h']),
                             int(template_rect['x']):
                             int(template_rect['x'] + template_rect['w'])]

            if 'template' in save_frames:
                cv2.imwrite(save_frames['template'], template)

            pf = filter_class(frame, template, **kwargs)

        # Process frame
        pf.process(frame)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            pf.render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            pf.render(frame_out)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))


def run_kalman_filter(kf, imgs_dir, noise, sensor, save_frames={},
                      template_loc=None):

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    frame_num = 0

    if sensor == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    elif sensor == "matching":
        frame = cv2.imread(os.path.join(imgs_dir, imgs_list[0]))
        template = frame[template_loc['y']:
                         template_loc['y'] + template_loc['h'],
                         template_loc['x']:
                         template_loc['x'] + template_loc['w']]

    else:
        raise ValueError("Unknown sensor name. Choose between 'hog' or "
                         "'matching'")

    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Sensor
        if sensor == "hog":
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                                    padding=(8, 8), scale=1.05)

            if len(weights) > 0:
                max_w_id = np.argmax(weights)
                z_x, z_y, z_w, z_h = rects[max_w_id]

                z_x += z_w // 2
                z_y += z_h // 2

                z_x += np.random.normal(0, noise['x'])
                z_y += np.random.normal(0, noise['y'])

        elif sensor == "matching":
            corr_map = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF)
            z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)

            z_w = template_loc['w']
            z_h = template_loc['h']

            z_x += z_w // 2 + np.random.normal(0, noise['x'])
            z_y += z_h // 2 + np.random.normal(0, noise['y'])

        x, y = kf.process(z_x, z_y)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            cv2.circle(out_frame, (int(z_x), int(z_y)), 20, (0, 0, 255), 2)
            cv2.circle(out_frame, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.rectangle(out_frame, (int(z_x) - z_w // 2, int(z_y) - z_h // 2),
                          (int(z_x) + z_w // 2, int(z_y) + z_h // 2),
                          (0, 0, 255), 2)

            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            cv2.circle(frame_out, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print('Working on frame {}'.format(frame_num))


def part_1b():
    print("Part 1b")

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    # Define process and measurement arrays if you want to use other than the
    # default. Pass them to KalmanFilter.
    Q = None  # Process noise array
    R = None  # Measurement noise array

    kf = ps5.KalmanFilter(template_loc['x'], template_loc['y'])

    save_frames = {10: os.path.join(output_dir, 'ps5-1-b-1.png'),
                   30: os.path.join(output_dir, 'ps5-1-b-2.png'),
                   59: os.path.join(output_dir, 'ps5-1-b-3.png'),
                   99: os.path.join(output_dir, 'ps5-1-b-4.png')}

    run_kalman_filter(kf,
                      os.path.join(input_dir, "circle"),
                      NOISE_2,
                      "matching",
                      save_frames,
                      template_loc)


def part_1c():
    print("Part 1c")

    init_pos = {'x': 311, 'y': 217}

    # Define process and measurement arrays if you want to use other than the
    # default. Pass them to KalmanFilter.
    Q = None  # Process noise array
    R = None  # Measurement noise array

    kf = ps5.KalmanFilter(init_pos['x'], init_pos['y'])

    save_frames = {10: os.path.join(output_dir, 'ps5-1-c-1.png'),
                   33: os.path.join(output_dir, 'ps5-1-c-2.png'),
                   84: os.path.join(output_dir, 'ps5-1-c-3.png'),
                   159: os.path.join(output_dir, 'ps5-1-c-4.png')}

    run_kalman_filter(kf,
                      os.path.join(input_dir, "walking"),
                      NOISE_1,
                      "hog",
                      save_frames)


def part_2a():

    template_loc = {'y': 72, 'x': 140, 'w': 50, 'h': 50}

    save_frames = {10: os.path.join(output_dir, 'ps5-2-a-1.png'),
                   30: os.path.join(output_dir, 'ps5-2-a-2.png'),
                   59: os.path.join(output_dir, 'ps5-2-a-3.png'),
                   99: os.path.join(output_dir, 'ps5-2-a-4.png')}

    num_particles = 0  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 8  # Define the value of sigma for the particles movement (dynamics)

    run_particle_filter(ps5.ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "circle"),
                        template_loc,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_mse,
                        sigma_dyn=sigma_dyn,
                        template_coords=template_loc)  # Add more if you need to


def part_2b():

    template_loc = {'x': 360, 'y': 141, 'w': 127, 'h': 179}
    box = {'x_min': 320, 'x_max': 450, 'y_min': 120, 'y_max': 320}

    save_frames = {10: os.path.join(output_dir, 'ps5-2-b-1.png'),
                   33: os.path.join(output_dir, 'ps5-2-b-2.png'),
                   84: os.path.join(output_dir, 'ps5-2-b-3.png'),
                   99: os.path.join(output_dir, 'ps5-2-b-4.png')}

    num_particles = 0  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)

    run_particle_filter(ps5.ParticleFilter,  # particle filter model class
                        os.path.join(input_dir, "pres_debate_noisy"),
                        template_loc,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_mse,
                        sigma_dyn=sigma_dyn,
                        template_coords=template_loc,
                        use_box_initialization=True, box=box)  # Add more if you need to


def part_3():
    template_rect = {'x': 538, 'y': 377, 'w': 73, 'h': 117}
    box = {'x_min': 500, 'x_max': 600, 'y_min': 350, 'y_max': 500}

    save_frames = {22: os.path.join(output_dir, 'ps5-3-a-1.png'),
                   50: os.path.join(output_dir, 'ps5-3-a-2.png'),
                   160: os.path.join(output_dir, 'ps5-3-a-3.png')}

    num_particles = 700  # Define the number of particles
    sigma_mse = 30  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 20  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.1  # Set a value for alpha

    run_particle_filter(ps5.AppearanceModelPF,  # particle filter model class
                        os.path.join(input_dir, "pres_debate"),
                        # input video
                        template_rect,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_mse,
                        sigma_dyn=sigma_dyn, alpha=alpha,
                        template_coords=template_rect, in_gray_mode=False, use_constant_alpha=False,
                        use_box_initialization=True, box=box)  # Add more if you need to


def part_4():
    template_rect = {'x': 210, 'y': 37, 'w': 103, 'h': 285}
    box = {'x_min': 200, 'x_max': 310, 'y_min': 30, 'y_max': 320}

    save_frames = {40: os.path.join(output_dir, 'ps5-4-a-1.png'),
                   100: os.path.join(output_dir, 'ps5-4-a-2.png'),
                   240: os.path.join(output_dir, 'ps5-4-a-3.png'),
                   300: os.path.join(output_dir, 'ps5-4-a-4.png')}

    num_particles = 100  # Define the number of particles
    sigma_md = 30  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 9  # Define the value of sigma for the particles movement (dynamics)

    run_particle_filter(ps5.MDParticleFilter,
                        os.path.join(input_dir, "pedestrians"),
                        template_rect,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_md,
                        sigma_dyn=sigma_dyn,
                        template_coords=template_rect,
                        use_box_initialization=True, box=box)


def part_5():
    """Tracking multiple Targets.

    Use either a Kalman or particle filter to track multiple targets
    as they move through the given video.  Use the sequence of images
    in the TUD-Campus directory.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    save_frames = {29: os.path.join(output_dir, 'ps5-5-a-1.png'),
                   56: os.path.join(output_dir, 'ps5-5-a-2.png'),
                   71: os.path.join(output_dir, 'ps5-5-a-3.png')}

    # kf = ps5.KalmanFilter(105, 260, R=0.2 * np.eye(2))
    # template_rect = {'x': 55, 'y': 155, 'w': 100, 'h': 215}
    #
    # run_kalman_filter(kf,
    #                   os.path.join(input_dir, "TUD-CAMPUS"),
    #                   NOISE_1,
    #                   "matching",
    #                   save_frames,
    #                   template_rect)

    # kf = ps5.KalmanFilter(10, 235, R=0.2 * np.eye(2))
    #
    # kalman.run_kalman_filter(kf,
    #                   os.path.join(input_dir, "TUD-CAMPUS"),
    #                   NOISE_1,
    #                   "matching",
    #                   save_frames,
    #                   None)

    # kf = ps5.KalmanFilter(300, 235, R=0.2 * np.eye(2))
    #
    # kalman_2.run_kalman_filter(kf,
    #                          os.path.join(input_dir, "TUD-CAMPUS"),
    #                          NOISE_1,
    #                          "matching",
    #                          save_frames,
    #                          None)
    # ------------------------------------ 15, 26

    num_particles_1 = 1300
    sigma_md_1 = 0
    sigma_dyn_1 = 18
    filter_1 = None
    box_1 = {'x_min': 290, 'x_max': 340, 'y_min': 175, 'y_max': 300}

    num_particles_2 = 600
    sigma_md_2 = 0
    sigma_dyn_2 = 17
    filter_2 = None
    box_2 = {'x_min': 65, 'x_max': 175, 'y_min': 125, 'y_max': 355}

    num_particles_3 = 800
    sigma_md_3 = 20
    sigma_dyn_3 = 20
    filter_3 = None
    box_3 = {'x_min': 0, 'x_max': 100, 'y_min': 125, 'y_max': 400}

    imgs_list = [f for f in os.listdir("input_images/TUD-Campus")
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    frame_num = 1
    template_1 = multi_filter.get_template_1()
    template_2 = multi_filter.get_template_2()
    template_3 = multi_filter.get_template_3()

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join("input_images/TUD-Campus", img))
        if filter_1 is None:
            filter_1 = ps5.AppearanceModelPF(frame, template_1, num_particles=num_particles_1, sigma_exp=sigma_md_1,
                                             sigma_dyn=sigma_dyn_1, alpha=0.0, in_gray_mode=False,
                                             use_threshold=False, threshold=40.0,
                                             use_box_initialization=True, box=box_1)
        if filter_2 is None:
            filter_2 = ps5.AppearanceModelPF(frame, template_2, num_particles=num_particles_2, sigma_exp=sigma_md_2,
                                             sigma_dyn=sigma_dyn_2, alpha=0.0, in_gray_mode=False,
                                             use_box_initialization=True, box=box_2)
        if filter_3 is None:
            filter_3 = ps5.AppearanceModelPF(frame, template_3, num_particles=num_particles_3, sigma_exp=sigma_md_3,
                                             sigma_dyn=sigma_dyn_3, alpha=0.0, in_gray_mode=False,
                                             use_box_initialization=True, box=box_3)

        multi_filter.process_filters(filter_1, filter_2, filter_3, frame, frame_num, save_frames=save_frames)

        frame_num += 1
    # print("done")


def part_6():
    """Tracking pedestrians from a moving camera.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    template_rect = {'x': 96, 'y': 55, 'w': 30, 'h': 72}

    save_frames = {60: os.path.join(output_dir, 'ps5-6-a-1.png'),
                   160: os.path.join(output_dir, 'ps5-6-a-2.png'),
                   186: os.path.join(output_dir, 'ps5-6-a-3.png')}

    num_particles = 300
    sigma_md = 10
    sigma_dyn = 17
    box = {'x_min': 85, 'x_max': 121, 'y_min': 50, 'y_max': 150}

    run_particle_filter(ps5.MDParticleFilter,
                        os.path.join(input_dir, "follow"),
                        template_rect,
                        save_frames,
                        num_particles=num_particles, sigma_exp=sigma_md,
                        sigma_dyn=sigma_dyn, in_gray_mode=False, alpha=0.5,
                        template_coords=template_rect, min_d=-0.3, max_d=0.3,
                        use_box_initialization=True, box=box,
                        use_alpha_blending=True)
    print("done")

if __name__ == '__main__':
    # part_1b()
    # part_1c()
    # part_2a()
    # part_2b()
    # part_3()
    part_4()
    # part_5()
    # part_6()
