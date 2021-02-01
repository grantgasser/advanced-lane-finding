"""
Perform camera calibration using black/white chessboard images

Reference: https://docs.opencv.org/4.4.0/dc/dbb/tutorial_py_calibration.html
"""
import glob

import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def calibrate(path, xy, draw_corners=False):
    """
    Perform camera calibration using black/white chessboard images

    Args:
        path (str): path of images in glob format
        xy ((int, int)): tuple of ints, number of x and y corners
        draw_corners (bool): whether to draw and plot corners of each chessboard

    Returns:
        calibration information (camera matrix, distortion coefficients, vectors)
    """
    images = glob.glob(path)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((xy[0]*xy[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:xy[0],0:xy[1]].T.reshape(-1, 2)  # keep z=0 for all
    #print(objp)  # [[0, 0, 0], [1, 0, 0], [2, 0, 0] .... [0, 1, 0], [1, 1, 0], [2, 1, 0]...]

    for img_file in images:
        # read and convert to gray
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, xy, None)

        # if found
        if ret == True:
            objpoints.append(objp)  # 3d obj points
            imgpoints.append(corners)  # 2d img points

            # drawing can help to validate its working
            if draw_corners:
                cv2.drawChessboardCorners(img, xy, corners, ret)
                plt.imshow(img)
                plt.show()

    # calibrate the camera once we've gotten all the obj and img pts
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, 
        imgpoints, 
        gray.shape[::-1],
        None, 
        None
    )

    return ret, mtx, dist, rvecs, tvecs


def undistort(img, mtx, dist):
    """
    Given distortion coefficients, undistort the img

    Args:
        img (np.ndarray): img to be undistorted
        mtx (3x3 np.ndarray): camera projection matrix
        dist (5x1 np.ndarray): distortion coefficients (k1, k2, p1, p2, k3)
    
    Returns:
        undistorted image as np.ndarray
    """
    return cv2.undistort(img, mtx, dist, None, mtx) 


def test_calibrate():
    # test camera calibration
    ret, mtx, dist, rvecs, tvecs = calibrate(
        path='../camera_cal/calibration*.jpg', 
        xy=(9, 6),
        draw_corners=False
    )

    # undistort a test image
    test_img = cv2.imread('../camera_cal/calibration3.jpg')
    undist = undistort(test_img, mtx, dist)

    # plot before and after correcting distortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(test_img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig('../output_images/test_undistort.png')
    plt.show()


# test_calibrate()