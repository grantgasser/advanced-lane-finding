# Advanced Lane Finding using Computer Vision

This is a detailed description of a software pipeline that identifies lane lines in a video stream. See `src/main.py` for the main script that contains the pipeline.

# Project Output Slice
[![Output Gif](output_videos/project_video.gif)](https://youtu.be/Z7qNvgJ7ehc)

# Steps Explained
## Camera Calibration
First, I performed [camera calibration](https://docs.opencv.org/4.4.0/dc/dbb/tutorial_py_calibration.html) using chessboard images stored in `camera_cal/`:


![Undistorted Image](./output_images/test_undistort.png)

See `src/calibrate.py` for the code.

## Gradient and Color Thresholding
Next, I performed thresholding using a combination of the x-gradient (using Sobel), the direction of the gradient, and the saturation of the image. Using saturation, or the S channel of HLS, is robust to variations in lighting. 

Note that using "AND", `&`, for combining the color and gradient threshold returns essentially _just_ the lane lines:

![Thresholded AND Image](./output_images/test_threshold_AND.png)

This might be useful, but we likely need more of the lanes, even if other elements of the image are picked up as well (we can mask later). Here is what it looks like using "OR", `|`, when combining: 

![Thresholded OR Image](./output_images/test_threshold_OR.png)

Here, the lane lines show up clearly. See `src/threshold.py` for the code.

## Perspective Transform (Bird's-eye view)
The next thing to do is to measure curvature. This is best done by looking at the lanes from a top-down view. Using a template image w/ relatively straight lane lines (`test_images/straight_lines1.png`), I performed a perspective transform using 4 manually derived source points from the image and mapping them to a warped, transformed image.

Here is the original image with the points (and lines) drawn: 
![Transform Points](./output_images/transform_pts.png)

and here is the transformed image w/ the associated points: 
![Transformed Image](./output_images/transform_img.png)

Here is what the transform looks like on an undistorted and thresholded image of a _curved_ road: 

![Original Binary](./output_images/original_binary.png)


![Test Transform](./output_images/test_transform.png)

Here, one can clearly see the right curvature of the road. Fortunately, despite the curvature, the lanes are still parallel, indicating the transform was done correctly. See `src/perspective_transform.py` for the code.

## Lane Identification and Fitting
TODO: add details
### Histogram Peaks
![Histogram](./output_images/histogram.png)
### Sliding Window Approach
![Sliding Windows](./output_images/sliding_windows.png)
#### Lane Lines Fit
![Lane Lines Fit](./output_images/lane_lines_fit.png)
