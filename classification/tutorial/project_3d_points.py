import numpy as np
import cv2


def main():
    src = np.ones((6, 3))
    src[:,1] = 2
    src[:,2] = range(6) # source points
    rvec = np.array([0,0,0], np.float) # rotation vector
    tvec = np.array([0,0,0], np.float) # translation vector
    fx = fy = 1.0
    cx = cy = 0.0
    camera_matrix = np.array([[fx,0,cx], [0,fy,cy], [0,0,1]])
    result = cv2.projectPoints(src, rvec, tvec, camera_matrix, None)
    for n in range(len(src)):
        print(src[n], '==>', result[0][n])


if __name__ == "__main__":
    main()