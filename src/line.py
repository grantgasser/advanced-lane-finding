"""
After preprocessing frames, find lines and store information from previous frames
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Line(object):
    def __init__(self):
        pass

    def find_line(self, processed_frame):
        """
        Find lane lines in a new, pre-processed frame
        """
        # start looking in bottom half, lanes tend to be straighter closer to car
        bottom_half = processed_frame[processed_frame.shape[0]//2:, :]
        hist = np.sum(bottom_half, axis=0)

        # image for drawing on
        draw_img = np.dstack((processed_frame, processed_frame, processed_frame))*255

        # get peak for left and right lane
        midpoint = bottom_half.shape[1] // 2
        leftx_base = np.argmax(hist[:midpoint])
        rightx_base = np.argmax(hist[midpoint:]) + midpoint

        # prepare sliding window search
        # hyperparameters
        n_windows = 9  # num vertical windows in image
        margin = 100  # +/- margin of windows
        min_pix = 50  # min nonzero pixels found in window to re-center the window

        window_height = np.int(processed_frame.shape[0]/n_windows)
        nonzeroy, nonzerox = processed_frame.nonzero()  # find x, y positions of nonero pixels
        
        # current positions of windows
        leftx_curr = leftx_base  
        rightx_curr = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        # sliding windows
        for window in range(n_windows):

            # determine window locations
            bottomy = processed_frame.shape[0] - (window * window_height)
            topy = processed_frame.shape[0] - ((window+1) * window_height)
            left_window_leftx = leftx_curr - margin
            left_window_rightx = leftx_curr + margin
            right_window_leftx = rightx_curr - margin
            right_window_rightx = rightx_curr + margin

            # draw left and right windows given (x,y) corners of rectangle
            cv2.rectangle(
                draw_img, 
                (left_window_leftx, topy), 
                (left_window_rightx, bottomy), 
                (0, 255, 0), 
                2
            )
            cv2.rectangle(
                draw_img, 
                (right_window_leftx, topy), 
                (right_window_rightx, bottomy), 
                (0, 255, 0), 
                2
            )

            # add pixel indices that are within the window 
            inside_left_inds = ((nonzerox > left_window_leftx) & (nonzerox < left_window_rightx) & \
                (nonzeroy > topy) & (nonzeroy < bottomy)).nonzero()[0]
            inside_right_inds = ((nonzerox > right_window_leftx) & (nonzerox < right_window_rightx) & \
                (nonzeroy > topy) & (nonzeroy < bottomy)).nonzero()[0]

            left_lane_inds.append(inside_left_inds)
            right_lane_inds.append(inside_right_inds)

            # => adjust prediction of lane side to side by re-centering window to mean of pixels
            if len(inside_left_inds) > min_pix:
                leftx_curr = np.int(np.mean(nonzerox[inside_left_inds]))
            if len(inside_right_inds) > min_pix:
                rightx_curr = np.int(np.mean(nonzerox[inside_right_inds]))

        
        # combine ALL pixel indexes from ALL windows of a given lane into one index list
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # get ALL left and right lane pixel positions using the indexes
        # the example has len(nonzerox) == 67000 and len(left_lane_inds) == 37000
        # makes sense that roughly 50% of nonzero pixels are in the left lane
        leftx = nonzerox[left_lane_inds]  # which nonzero pixels are within the left lane window?
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        return leftx, lefty, rightx, righty, draw_img


    def fit_polynomials(self, processed_frame):
        """
        Fit a 2nd degree polynomial to the chosen pixels of the lane.

        We have left and right lanes.
        """
        leftx, lefty, rightx, righty, draw_img = self.find_line(processed_frame)

        # get fit coefficients, note x and y are reversed from the norm
        left_coef = np.polyfit(lefty, leftx, 2)
        right_coef = np.polyfit(righty, rightx, 2)

        # ax^2 + bx + c or ay^2 + by + c
        y = np.linspace(0, processed_frame.shape[0]-1, processed_frame.shape[0])
        left_fit = left_coef[0]*y**2 + left_coef[1]*y + left_coef[2]
        right_fit = right_coef[0]*y**2 + right_coef[1]*y + right_coef[2]

        # change color of lane pixels
        draw_img[lefty, leftx] = [255, 0, 0]
        draw_img[righty, rightx] = [0, 0, 255]

        # plot the fit on the lane line image
        # plt.imshow(draw_img)
        # plt.plot(left_fit, y, color='yellow')
        # plt.plot(right_fit, y, color='yellow')
        # plt.title('Lane Lines Fit')
        # plt.savefig('../output_images/lane_lines_fit.png')
        # plt.show()

        return left_fit, right_fit, y



        
