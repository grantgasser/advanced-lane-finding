"""
After preprocessing frames, find lines and store information from previous frames
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

from collections import deque

class Lane(object):
    def __init__(self):
        self.detected = True  # did last frame pass sanity check?
        self.consecutive_bad_frames = 0  # num of consecutive bad frames (failed sanity check)
        self.num_prev_frames = 3  # num prev frames to store data for

        # TODO: finish attributes
        self.recent_leftx = deque(maxlen=self.num_prev_frames)
        self.recent_lefty = deque(maxlen=self.num_prev_frames)
        self.recent_rightx = deque(maxlen=self.num_prev_frames)
        self.recent_righty = deque(maxlen=self.num_prev_frames)
        self.left_fit = deque(maxlen=self.num_prev_frames)
        self.right_fit = deque(maxlen=self.num_prev_frames)
        self.offset = None
        self.radius_curve_left = None
        self.radius_curve_right = None
        self.lane_width = 600  # near bottom of image, lane is typically ~600 pixels

    def window_search(self, processed_frame):
        """
        Find lane lines in a transformed frame using the window search
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
        nonzeroy, nonzerox = processed_frame.nonzero()  # find x, y positions of nonzero pixels
        
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

            # TODO: handle sharp turns where lane goes off of image (avoid stacking windows on side of img to top)
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


    def margin_search(processed_frame, margin=100):
        """
        Find lane lines using margin search of prior lane
        """
        # find x, y positions of nonzero pixels
        nonzeroy, nonzerox = processed_frame.nonzero()

        # image for drawing on
        draw_img = np.dstack((processed_frame, processed_frame, processed_frame))*255

        # get indexes of these pixels if they fit within prev lane line +/- margin
        left_lane_inds = ((nonzerox > (self.left_fit[-1][0]*(nonzeroy**2) + self.left_fit[-1][1]*nonzeroy + 
                    self.left_fit[-1][2] - margin)) & (nonzerox < (self.left_fit[-1][0]*(nonzeroy**2) + 
                    self.left_fit[-1][1]*nonzeroy + self.left_fit[-1][2] + margin)))
        right_lane_inds = ((nonzerox > (self.right_fit[-1][0]*(nonzeroy**2) + self.right_fit[-1][1]*nonzeroy + 
                    self.right_fit[-1][2] - margin)) & (nonzerox < (self.right_fit[-1][0]*(nonzeroy**2) + 
                    self.right_fit[-1][1]*nonzeroy + self.right_fit[-1][2] + margin)))

        # TODO: also need to address when line goes off of frame on sharp turn

        # get all left and right lane pixel positions for entire frame (w/in margin)
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, draw_img


    def fit_polynomials(self, processed_frame):
        """
        Fit a 2nd degree polynomial to the chosen pixels of the lane.

        We have left and right lanes.
        """
        # determine what kind of search to do
        if not self.detected:
            if self.consecutive_bad_frames >= 2:
                # reset and do window search from scratch
                leftx, lefty, rightx, righty, draw_img = self.window_search(processed_frame)
            else:
                # TODO: skip back once to get previous of previous 
                
        else:
            # margin search (more efficient)
            leftx, lefty, rightx, righty, draw_img = self.margin_search(processed_frame)

        # store x lane values
        self.recent_leftx.append(leftx)
        self.recent_lefty.append(lefty)
        self.recent_rightx.append(rightx)
        self.recent_righty.append(righty)

        # get fit coefficients, note x and y are reversed from the norm
        left_coef = np.polyfit(lefty, leftx, 2)
        right_coef = np.polyfit(righty, rightx, 2)

        # ay^2 + by + c
        y = np.linspace(0, processed_frame.shape[0]-1, processed_frame.shape[0])
        left_fit = left_coef[0]*y**2 + left_coef[1]*y + left_coef[2]
        right_fit = right_coef[0]*y**2 + right_coef[1]*y + right_coef[2]

        # store coefficients of fit
        self.left_fit.append(left_fit)
        self.right_fit.append(right_fit)

        # change color of lane pixels
        draw_img[lefty, leftx] = [255, 0, 0]
        draw_img[righty, rightx] = [0, 0, 255]

        # pixels to meters conversion
        y_meters_per_px = 30/draw_img.shape[0]  # lane is roughly 30m long in the images
        x_meteres_per_px = 3.7/self.avg_lane_pixel_length  # lane is roughly 3.7m (per US regulation), tends to cover abt 600 pixels in x direction

        # find offset
        mid_lane = int(left_fit[-1] + ((right_fit[-1] - left_fit[-1]) // 2))
        offset = int(mid_lane - (draw_img.shape[1] // 2))
        offset_meters = offset * x_meteres_per_px

        # TODO: measure curvature (see radius of curvature formula online)
        # y_bottom = np.max(y)  # want curvature at bottom of image, closest to car
        # left_curve_radius = ((1 + (2*left_fit[0]*y_bottom*y_meters_per_px + left_fit[1])**2)**1.5) / np.abs(2*left_fit[0])
        # right_curve_radius = ((1 + (2*right_fit[0]*y_bottom*y_meters_per_px + right_fit[1])**2)**1.5) / np.abs(2*right_fit[0])
        # print('Left curve radius:', left_curve_radius)
        # print('Right curve radius:', right_curve_radius)

        # perform sanity check on lane predictions
        self.sanity_check()

        # plot the fit on the lane line image
        # plt.imshow(draw_img)
        # plt.plot(left_fit, y, color='yellow')
        # plt.plot(right_fit, y, color='yellow')
        # plt.title('Lane Lines Fit')
        # # plt.savefig('../output_images/lane_lines_fit.png')
        # plt.show()

        return left_fit, right_fit, y, offset_meters


    def sanity_check():
        """Verify lane predictions make reasonable sense"""
        pass_tests = True

        # TODO: should have "similar" curvature
        
        # TODO: should be separated by approx. correct horizontal distance
        width = np.average(self.recent_rightx[-1]) - np.average(self.recent_leftx[-1])
        if width < (self.lane_width - 100) or width > (self.lane_width + 100):
            pass_tests = False
            print(f'lane width fail = {estimated_lane_width}. resetting...')

        # TODO: should be roughly parallel


        if pass_tests:
            self.detected = True
            self.consecutive_bad_frames = 0
        else:
            self.detected = False
            self.consecutive_bad_frames += 1
    
    def smooth(n=3):
        """Average over last n frames"""
        raise NotImplementedError()



        
