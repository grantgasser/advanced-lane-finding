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
        plt.plot(hist)
        plt.title('Histogram')
        plt.savefig('../output_images/histogram.png')
        plt.show()

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

            print('bottomy:', bottomy, 'topy:', topy)
            print('left_window_leftx:', left_window_leftx, 'left_window_rightx:', left_window_rightx)

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
            print('nonzerox:', nonzerox)
            print('nonzeroy:', nonzeroy)
            # add pixel indices that are within the window 
            inside_left_inds = ((nonzerox > left_window_leftx) & (nonzerox < left_window_rightx) & \
                (nonzeroy > topy) & (nonzeroy < bottomy)).nonzero()[0]
            inside_right_inds = ((nonzerox > right_window_leftx) & (nonzerox < right_window_rightx) & \
                (nonzeroy > topy) & (nonzeroy < bottomy)).nonzero()[0]

            print('inside left inds:', inside_left_inds)
            left_lane_inds.append(inside_left_inds)
            right_lane_inds.append(inside_right_inds)

            # if num pixels in window > min_pix: re-center window (leftx_curr, rightx_curr)
            if len(inside_left_inds) > min_pix:
                leftx_curr = np.int(np.mean(nonzerox[inside_left_inds]))
            if len(inside_right_inds) > min_pix:
                rightx_curr = np.int(np.mean(nonzerox[inside_right_inds]))

        
        # now that we have our windows tracking both lane lines
        # combine ALL pixels in a given lane into one index list
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # TODO: extract left and right lane pixel positions
        # and return

        
        # plt.imshow(draw_img)
        # plt.title('Sliding Windows')
        # plt.savefig('../output_images/sliding_windows2.png')
        # plt.show()


    def fit_polynomials(self, processed_frame):
        pass





        
