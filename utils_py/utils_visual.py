# loading library
import cv2
import numpy as np
import os


def motion_blur(img, kernel_size=15, vertical=True):
    '''
    reference: https://www.geeksforgeeks.org/opencv-motion-blur-in-python/
    :param img: cv2 image
    :param kernel_size: the blur kernel size
    :param vertical: do vertical blur or horizontal blur
    :return: motion blured img
    '''

    # Create the kernel.
    kernel = np.zeros((kernel_size, kernel_size))

    # Fill the middle row with ones.
    if vertical:
        kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    else:
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    # Normalize.
    kernel /= kernel_size

    # Apply the vertical kernel.
    mb_img = cv2.filter2D(img, -1, kernel)

    return mb_img


def gaussian_blur(img, kernel_size=7, sigma=3):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def blur_image(cv2_img, blur_type):
    if blur_type == 0:
        return cv2_img
    elif blur_type == 1:
        blur_img = gaussian_blur(cv2_img, kernel_size=7, sigma=3)
    elif blur_type == 2:
        blur_img = motion_blur(cv2_img, kernel_size=20, vertical=True)
    else:
        blur_img = motion_blur(cv2_img, kernel_size=25, vertical=False)
    return blur_img


if __name__ == '__main__':

    image_path = '../tmpdata/0009050.jpg'
    image_path = '../tmpdata/0003300.jpg'
    img = cv2.imread(image_path)

    blur_image = motion_blur(img, kernel_size=20, vertical=True)
    cv2.imshow('blur image', blur_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


