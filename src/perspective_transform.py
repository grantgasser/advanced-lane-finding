"""
Perform perspective transform on an image

Assumption (may have to be addressed later): road is flat plane
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_plot_save(img, points, plt_title=None, save_path=None):
    """
    Draw lines on an image from point a to point b

    Args:
        img (np.ndarray): BGR image to draw points and lines on
        points (np.ndarry): 4x2 array containing points in following order:
            bottom_left, top_left, top_right, bottom_right
        plt_title (str): title of the plot
        save_path (str): where to save the plot as png file
    """
    # draw
    line_color = (255, 0, 0)
    img = cv2.line(img, tuple(points[0]), tuple(points[1]), line_color, 3)
    img = cv2.line(img, tuple(points[2]), tuple(points[3]), line_color, 3)

    # plot
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(points[0][0], points[0][1], '.')
    plt.plot(points[1][0], points[1][1], '.')
    plt.plot(points[2][0], points[2][1], '.')
    plt.plot(points[3][0], points[3][1], '.')
    if plt_title:
        plt.title(plt_title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# use straight lines template to perform perspective transform and get M
img = cv2.imread('../test_images/straight_lines1.jpg')
draw_img = np.copy(img)

# eyeballing 4 corners of the lane
bottom_left = (200, img.shape[0])
top_left = (595, 450)
top_right = (685, 450)
bottom_right = (1110, img.shape[0])

# source points in original image
src = np.float32(
    [bottom_left,
    top_left,
    top_right,
    bottom_right]
)

draw_plot_save(
    img=draw_img, 
    points=src, 
    plt_title='Perspective Transform Pts', 
    save_path='../output_images/transform_pts.png'
)


# destination pts
dst_bottom_left = (350, img.shape[0])
dst_top_left = (350, 0)
dst_top_right = (950, 0)
dst_bottom_right = (950, img.shape[0])

# dest points in warped image, in same order as original
dst = np.float32(
    [dst_bottom_left,
    dst_top_left,
    dst_top_right,
    dst_bottom_right]
)

# get transformation matrix and perform transform
M = cv2.getPerspectiveTransform(src, dst)
size = (img.shape[1], img.shape[0])
transformed_img = cv2.warpPerspective(img, M, size)

draw_plot_save(
    img=transformed_img, 
    points=dst, 
    plt_title='Transformed Image', 
    save_path='../output_images/transform_img.png'
)

# draw lines on transformed image
line_color = (255, 0, 0)
lines_img = cv2.line(transformed_img, dst_bottom_left, dst_top_left, line_color, 3)
lines_img = cv2.line(transformed_img, dst_bottom_right, dst_top_right, line_color, 3)

plt.imshow(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
plt.plot(dst_bottom_left[0], dst_bottom_left[1], '.')
plt.plot(dst_top_left[0], dst_top_left[1], '.')
plt.plot(dst_top_right[0], dst_top_right[1], '.')
plt.plot(dst_bottom_right[0], dst_bottom_right[1], '.')
plt.title('Transformed Image')
plt.savefig('../output_images/transform_img.png')
plt.show()

def get_transform_matrix():
    return src, dst, M