"""
Lane-finding pipeline
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibrate import calibrate, undistort
import threshold as th

# calibrate the camera using the given chessboard images
ret, mtx, dist, rvecs, tvecs = calibrate(
    path='../camera_cal/calibration*.jpg', 
    xy=(9, 6),
    draw_corners=False
)

# undistort an image
bgr_img = cv2.imread('../test_images/test6.jpg')
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)  # BGR => RGB
undist = undistort(rgb_img, mtx, dist)
plt.imshow(undist)
plt.title('Undistorted')
plt.show()

# convert to gray
gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)  # RGB => GRAY

# apply gradient and color thresholding
gradx = th.abs_sobel_thresh(gray)
direction = th.dir_thresh(gray)
gradient_binary = np.zeros_like(direction)
gradient_binary[(gradx == 1) & (direction == 1)] = 1

color_binary = th.saturation_thresh(bgr_img)

# combine gradient and color thresholding
final_output = th.threshold(gradient_binary, color_binary)
plt.imshow(final_output, cmap='gray')
plt.title('Thresholding')
plt.show()

# perspective transform: easier to measure curvature of lane from bird's eye view
# also makes it easier to match car's location with a road map
