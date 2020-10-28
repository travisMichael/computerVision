import cv2
import ps5
import os
import numpy as np
import multi_filter


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
        # template = frame[template_loc['y']:
        #                  template_loc['y'] + template_loc['h'],
        #            template_loc['x']:
        #            template_loc['x'] + template_loc['w']]
        template = multi_filter.get_template_3()

    else:
        raise ValueError("Unknown sensor name. Choose between 'hog' or "
                         "'matching'")

    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))
        if frame_num < 25:
            frame_num += 1
            continue

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

            z_w = template.shape[1]
            z_h = template.shape[0]

            z_x += z_w // 2 + np.random.normal(0, noise['x'])
            z_y += z_h // 2 + np.random.normal(0, noise['y'])

        x, y = kf.process(z_x, z_y)
        if frame_num == 36:
            x_int = int(x) - 15
            y_int = int(y)
            h_2 = int(template.shape[0]/2)
            w_2 = int(template.shape[1]/2) + 20
            template = frame[y_int-h_2:y_int + h_2+1, x_int-w_2:x_int + w_2]
            cv2.imwrite("out/new_template.png", template)

        if frame_num == 55:
            x_int = int(x)+5
            y_int = int(y) + 10
            h_2 = int(template.shape[0]/2) + 20
            w_2 = int(template.shape[1]/2)
            template = frame[y_int-h_2:y_int + h_2+1, x_int-w_2:x_int + w_2]
            cv2.imwrite("out/new_template.png", template)

        if frame_num == 65:
            x_int = int(x)
            y_int = int(y)
            h_2 = int(template.shape[0]/2)
            w_2 = int(template.shape[1]/2)
            template = frame[y_int-h_2:y_int + h_2+1, x_int-w_2:x_int + w_2]
            cv2.imwrite("out/new_template.png", template)


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
