import cv2
import ps5
import os
import numpy as np


def process_filters(filter_1, filter_2, filter_3, frame, frame_num, save_frames={}):
    out_frame = frame.copy()
    # save_frame = frame.copy()
    if frame_num < 49:
        filter_1.process(frame)
        filter_1.render(out_frame)

    if frame_num < 62:
        filter_2.process(frame)
        filter_2.render(out_frame)

    # if frame_num > 21:
    #     x, y = filter_3.process(frame)
    #     filter_3.render(out_frame)
        # render(frame_num, out_frame, filter_3, save_frames,  x, y, 3)

    # Update frame number
    # render_t()
    render(out_frame)
    save(frame_num, out_frame, save_frames)
    frame_num += 1
    if frame_num % 20 == 0:
        print('Working on frame {}'.format(frame_num))


def save(frame_num, frame, save_frames):
    # Render and save output, if indicated
    if frame_num in save_frames:

        cv2.imwrite(save_frames[frame_num], frame)


def render(frame):
    if True:  # For debugging, it displays every frame
        cv2.imshow('Tracking', frame)
        cv2.waitKey(1)

    # render_t(frame_num, x, y, filter, frame, f)


def render_t(n, x, y, filter, frame, f):
    template = filter.template
    if n == 54 and f == 3:
        x_int = int(x) - 10
        y_int = int(y)
        h_2 = int(template.shape[0]/2)
        w_2 = int(template.shape[1]/2)
        template = frame[y_int-h_2:y_int + h_2+1, x_int-w_2:x_int + w_2]
        filter.template = template
        cv2.imwrite("out/n_template.png", template)


def initialize_filter(filter_class, frame, template, **kwargs):

    # template = frame[int(template_rect['y']):int(template_rect['y'] + template_rect['h']),
    #            int(template_rect['x']):int(template_rect['x'] + template_rect['w'])]
    #
    # cv2.imwrite("out/template.png", template)

    pf = filter_class(frame, template, **kwargs)

    return pf

# filter_1 = multi_filter.initialize_filter(ps5.AppearanceModelPF, frame,
#                                           num_particles=num_particles_1, sigma_exp=sigma_md_1,
#                                           sigma_dyn=sigma_dyn_1, alpha=0.0)

def get_template_1():

    frame = cv2.imread("input_images/TUD-Campus/000001.jpg")
    # template_rect = {'x': 244, 'y': 190, 'w': 47, 'h': 135} # 09
    # template_rect = {'x': 162, 'y': 180, 'w': 47, 'h': 135} # 22
    template_rect = {'x': 300, 'y': 200, 'w': 25, 'h': 135}

    template = frame[int(template_rect['y']):int(template_rect['y'] + template_rect['h']),
               int(template_rect['x']):int(template_rect['x'] + template_rect['w'])]

    cv2.imwrite("out/template_1.png", template)

    return template


def get_template_2():

    # frame = cv2.imread("input_images/TUD-Campus/000001.jpg")
    # template_rect = {'x': 55, 'y': 175, 'w': 100, 'h': 145}
    frame = cv2.imread("input_images/TUD-Campus/000025.jpg")
    template_rect = {'x': 265, 'y': 195, 'w': 75, 'h': 145}

    template = frame[int(template_rect['y']):int(template_rect['y'] + template_rect['h']),
               int(template_rect['x']):int(template_rect['x'] + template_rect['w'])]

    cv2.imwrite("out/template_2.png", template)

    return template

# def get_template_3():
#
#     frame = cv2.imread("input_images/TUD-Campus/000025.jpg")
#     template_rect = {'x': 0, 'y': 175, 'w': 38, 'h': 145}
#
#     template = frame[int(template_rect['y']):int(template_rect['y'] + template_rect['h']),
#                int(template_rect['x']):int(template_rect['x'] + template_rect['w'])]
#
#     cv2.imwrite("out/template_3.png", template)
#
#     return template


def get_template_3():

    frame = cv2.imread("input_images/TUD-Campus/000022.jpg")
    template_rect = {'x': 0, 'y': 185, 'w': 38, 'h': 165}

    template = frame[int(template_rect['y']):int(template_rect['y'] + template_rect['h']),
               int(template_rect['x']):int(template_rect['x'] + template_rect['w'])]

    cv2.imwrite("out/template_3.png", template)

    return template

get_template_3()
print()