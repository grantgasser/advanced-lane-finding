"""
Apply gradient and color thresholding

Trying to detect lane lines of different colors and under varying degrees
or daylight and shadow (brightness)
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Gradient:

Sobel gives the gradient in the x and y direction. The x direction will likely
be the most useful. Magnitude and direction of gradient can also be used.
"""
def abs_sobel_thresh(img, orient='x', kernel=3, thresh=(30, 100)):
    """
    Apply Sobel x or Sobel y. Take abs value and threshold.
    
    Args:
        img (np.ndarray): grayscale image
        orient (str): 'x' or 'y' for direction of gradient
        kernel (int): size of Sobel kernel, e.g. (3,3), (5,5), etc.
        thresh ((int, int)): tuple of ints, lower thresh and upper thresh

    Returns:
        binary (np.ndarray): thresholded binary image
    """
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)

    # abs and scale to 0-255
    abs_sobel = np.abs(sobel)
    scaled_sobel = np.uint8(255 * (abs_sobel/np.max(abs_sobel)))

    # threshold
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary


def dir_thresh(img, kernel=3, thresh=(0.85, 1.15)):
    """
    Threshold on the direction of the gradient
    
    Args:
        img (np.ndarray): grayscale image
        kernel (int): size of Sobel kernel, e.g. (3,3), (5,5), etc.
        thresh ((int, int)): tuple of ints, lower thresh and upper thresh

    Returns:
        binary (np.ndarray): thresholded binary image 
    """
    # get x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)

    # get direction of gradients
    direction = np.arctan2(np.abs(sobely), np.abs(sobelx))

    # threshold
    binary = np.zeros_like(direction)
    binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

    return binary


def mag_thresh(img, kernel=3, thresh=(30, 100)):
    """
    Apply gradient magnitude thresholding.

    Args:
        img (np.ndarray): grayscale image
        kernel (int): size of Sobel kernel, e.g. (3,3), (5,5), etc.
        thresh ((int, int)): tuple of ints, lower thresh and upper thresh

    Returns:
        binary (np.ndarray): thresholded binary image
    """
    # get x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)

    # get magnitude
    mag = np.sqrt(sobelx**2 + sobely**2)
    scaled_mag = np.uint8(255 * (mag/np.max(mag)))

    # threshold
    binary = np.zeros_like(scaled_mag)
    binary[(scaled_mag >= thresh[0]) & (scaled_mag <= thresh[1])] = 1
    
    return binary


"""
Color: 

HLS (Hue, Saturation, Lightness): H and S channels stay fairly consistent 
in shadow or excessive brightness unlike RGB color space

R channel (in RGB) seems to do well with white lines while S channel
seems to do well with both colors (may be able to combine them)
"""
def saturation_thresh(img, thresh=(90, 255)):
    """
    Use the S channel of HLS

    Args:
        img (np.ndarray): BGR (read in w/ cv2.imread) image
        thresh ((int, int)): tuple of ints, lower thresh and upper thresh

    Returns:
        binary (np.ndarray): thresholded binary image
    """
    # get saturation
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s = hls[:, :, 2]

    # threshold
    binary = np.zeros_like(s)
    binary[(s >= thresh[0]) & (s <= thresh[1])] = 1

    return binary



def threshold(gradient_binary, color_binary):
    """
    Combine color and gradient thresholding
    """
    binary = np.zeros_like(gradient_binary)
    # switch between OR and AND
    binary[(gradient_binary == 1) | (color_binary == 1)] = 1
    return binary


def test_thresholding():
    """
    Test Gradient functions
    """
    img = cv2.imread('../test_images/test6.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # combine threshold functions (looks pretty decent)
    gradx = abs_sobel_thresh(gray)
    #grady = abs_sobel_thresh(gray, orient='y')
    direction = dir_thresh(gray)
    #mag = mag_thresh(gray, kernel=3, thresh=(70, 100))

    gradient_binary = np.zeros_like(direction)
    gradient_binary[(gradx == 1) & (direction == 1)] = 1

    plt.imshow(gradient_binary, cmap='gray')
    plt.show()

    """
    Test Color function
    """
    color_binary = saturation_thresh(img)
    plt.imshow(color_binary, cmap='gray')
    plt.show()

    """
    Test combination of gradient and color
    """
    final_output = threshold(gradient_binary, color_binary)

    # not bad, actually only has the lane lines w/o masking (maybe can take advantage of that)
    # could switch to "OR" when combining if want MORE pixels
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(final_output, cmap='gray')
    ax2.set_title('Thresholded (OR)', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig('../output_images/test_threshold_OR.png')
    plt.show()

#test_thresholding()