import cv2
import ps5
import os
import numpy as np


def process_filters(filter_1, filter_2, filter_3, frame, frame_num, save_frames={}):

    # if frame_num > 49:
    #     continue
    # filter_1.process(frame)
    # render(frame_num, frame, filter_1, save_frames)

    # filter_2.process(frame)
    # render(frame_num, frame, filter_2, save_frames)

    if frame_num > 25:
        render(frame_num, frame, filter_3, save_frames)
        filter_3.process(frame)
        render(frame_num, frame, filter_3, save_frames)

    # Update frame number
    frame_num += 1
    if frame_num % 20 == 0:
        print('Working on frame {}'.format(frame_num))


def render(frame_num, frame, filter, save_frames):
    if True:  # For debugging, it displays every frame
        out_frame = frame.copy()
        filter.render(out_frame)
        cv2.imshow('Tracking', out_frame)
        cv2.waitKey(1)

    # Render and save output, if indicated
    if frame_num in save_frames:
        frame_out = frame.copy()
        filter.render(frame_out)
        cv2.imwrite(save_frames[frame_num], frame_out)


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

def get_template_3():

    frame = cv2.imread("input_images/TUD-Campus/000025.jpg")
    template_rect = {'x': 0, 'y': 175, 'w': 38, 'h': 145}

    template = frame[int(template_rect['y']):int(template_rect['y'] + template_rect['h']),
               int(template_rect['x']):int(template_rect['x'] + template_rect['w'])]

    cv2.imwrite("out/template_3.png", template)

    return template

get_template_1()
print()