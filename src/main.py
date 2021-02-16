"""
Lane-finding pipeline
"""
import glob
import os
import cProfile

import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibrate import calibrate, undistort
import threshold as th
import perspective_transform as pt
from lane import Lane
import postprocessing as post


def main():
    # calibrate the camera using the given chessboard images
    ret, mtx, dist, rvecs, tvecs = calibrate(
        path='../camera_cal/calibration*.jpg', 
        xy=(9, 6),
        draw_corners=False
    )

    # inst. Lane object
    lane = Lane()

    # read video
    predicted_frames = []
    input_video = 'project_video.mp4'
    cap = cv2.VideoCapture(os.path.join('../input_videos/', input_video))
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print('Cant receive frame. Exiting..')
            break
        
        # undistort an image
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR => RGB
        undist = undistort(rgb_img, mtx, dist)

        # convert to gray
        gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)  # RGB => GRAY

        # apply gradient and color thresholding
        gradx = th.abs_sobel_thresh(gray)
        direction = th.dir_thresh(gray)
        gradient_binary = np.zeros_like(direction)
        gradient_binary[(gradx == 1) & (direction == 1)] = 1

        color_binary = th.saturation_thresh(frame)

        # combine gradient and color thresholding
        thresholded_img = th.threshold(gradient_binary, color_binary)

        # perspective transform: easier to measure curvature of lane from bird's eye view
        # also makes it easier to match car's location with a road map
        src, dst, M, M_inv = pt.get_transform_matrix()

        # transform image
        size = (thresholded_img.shape[1], thresholded_img.shape[0])
        transformed_img = cv2.warpPerspective(thresholded_img, M, size)

        # draw lines on transformed
        gray_transformed_img = np.uint8(transformed_img*255)
        bgr_transformed_img = cv2.cvtColor(gray_transformed_img, cv2.COLOR_GRAY2BGR)
        #pt.draw_plot_save(bgr_transformed_img, dst, 'Test Transformation', '../output_images/test_transform.png')

        # fit lines
        left_fit, right_fit, y, offset_meters = lane.fit_polynomials(transformed_img)

        # create blank for drawing lane lines
        zeros = np.zeros_like(transformed_img).astype(np.uint8)
        draw_img = np.dstack((zeros, zeros, zeros))
        
        # format points for fill poly
        pts_left = np.array([np.transpose(np.vstack([left_fit, y]))])  # [left_fit ... y]
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit, y])))])
        pts = np.hstack((pts_left, pts_right))  # [pts_left, pts_right]
        cv2.fillPoly(draw_img, np.int_([pts]), (0, 255, 0))

        # unwarp transformed image
        unwarped = cv2.warpPerspective(draw_img, M_inv, (gray.shape[1], gray.shape[0]))

        # combine lane drawing w/ original image
        final_image = cv2.addWeighted(undist, 1, unwarped, 0.25, 0)

        # add measurement data to frame
        offset_side = 'left' if offset_meters < 0 else 'right'
        final_image = cv2.putText(final_image, f'Offset: {abs(offset_meters):0.2f}m {offset_side} of center', (50, 50), 
            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # show predict
        cv2.imshow('frame', cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

        # store predicted frames
        predicted_frames.append(final_image)

        if cv2.waitKey(1) == ord('q'):
            break


    # release cap object
    cap.release()
    cv2.destroyAllWindows()

    # use predicted frames to convert back to video
    # video_frames = post.write_images(predicted_frames, '../video_frames/')
    # clip = post.make_video(video_frames, os.path.join('../output_videos/', input_video))
    # post.write_gif(
    #     clip=clip, 
    #     path=os.path.join('../output_videos/', input_video + '.gif'),
    #     sub_start=15,
    #     sub_end=25
    # )

cProfile.run('main()', sort='cumulative')