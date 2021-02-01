"""
Perform perspective transform on an image

Assumption (may have to be addressed later): road is flat plane
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# def transform(img):
#     """
#     Perform perspective transform on an image
    
#     Args:
#         img (np.ndarray): image to do the transform on
    
#     """
#     src = None
#     dst = None

#     M = cv2.getPerspectiveTransform(src, dst)

#     size = (img.shape[1], img.shape[0])
#     transformed_img = cv2.warpPerspective(img, M, size)

#     return transformed_img, M


# eyeballing it
# use straight lines template to perform perspective transform and get M
img = cv2.imread('../test_images/straight_lines1.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.plot(595, 450, '.')
plt.plot(685, 450, '.')
plt.plot(200, img.shape[0], '.')
plt.plot(1110, img.shape[0], '.')
plt.title('Perspective Tfm Pts')
plt.savefig('../output_images/transform_pts.png')
plt.show()

# source points in original image
src = np.float32(
    [[595, 450],
    [685, 450],
    [1110, img.shape[0]],
    [200, img.shape[0]]]
)

# dest points in warped image, ensure they're in same order
dst = np.float32(
    [[350, 0],
    [950, 0],
    [950, img.shape[0]],
    [350, img.shape[0]]]
)

M = cv2.getPerspectiveTransform(src, dst)
size = (img.shape[1], img.shape[0])
transformed_img = cv2.warpPerspective(img, M, size)
plt.imshow(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
plt.plot(350, 0, '.')
plt.plot(950, 0, '.')
plt.plot(950, img.shape[0], '.')
plt.plot(350, img.shape[0], '.')
plt.title('Transformed Image')
plt.savefig('../output_images/transform_img.png')
plt.show()