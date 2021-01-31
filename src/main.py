"""
Script to organize and run the main line finding pipeline here
"""
from calibrate import calibrate, undistort

# calibrate the camera using the given chessboard images
ret, mtx, dist, rvecs, tvecs = calibrate(
    path='../camera_cal/calibration*.jpg', 
    xy=(9, 6),
    draw_corners=False
)

# undistort an image
test_img = cv2.imread('../camera_cal/calibration3.jpg')
undist = undistort(test_img, mtx, dist)


# perspective transform: easier to measure curvature of lane from bird's eye view
# also makes it easier to match car's location with a road map
