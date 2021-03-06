import cv2 as cv

import numpy as np


def preprocess(img: np.ndarray, dsize: tuple = (800, 600), mode: str = 'gaussian') -> np.ndarray:
    """
    Contains everything that needs to be done before the methods
    :param img: image in np.ndarray format
    :param dsize: desired size default 800 600
    :param mode: Modes can be defined
    :return: Preprocessed image
    """
    image = cv.resize(img, dsize)
    # image = blur_image(image,mode)
    return image


def blur_image(img: np.ndarray, mode: str) -> np.ndarray or None:
    """
    Blurr the image with a Gaussian filter
    :param img: Image in np.ndarray
    :param mode: what type of blur is needed
    :return: blured image
    """
    if mode == 'gaussian':
        sigma = img.shape[1] / 50
        return cv.GaussianBlur(img, (5, 5), sigma)
    else:
        print('Invalid mode')
        return None


def visualise_canny_thresholds(img: np.ndarray):
    """
    Helper function to visualize the effect of the threshold values in the canny edge detector
    :param img: Image in np.ndarray
    :return: Binary edge image (output of the canny edge detector)
    """
    # Choosing the default values and the font to print it on
    min_val = 0
    max_val = 255
    font = cv.FONT_HERSHEY_SIMPLEX

    while True:
        # Run the edge detector with the threshold values
        edge_image = cv.Canny(img, min_val, max_val, L2gradient=True)
        # Print the values on the image
        cv.putText(edge_image, 'min' + str(min_val) + '   max' + str(max_val), (30, 30), font, 1, (255, 255, 255), 2,
                   cv.LINE_AA)
        # Display it
        cv.imshow('Display', edge_image)
        # Wait for the input
        k = cv.waitKey(0)
        # Based on the input change the values or exit
        if k == ord('o'):
            min_val += 10
        elif k == ord('O'):
            min_val -= 10
        elif k == ord('l'):
            max_val -= 10
        elif k == ord('L'):
            max_val += 10
        elif k == ord('j'):
            break
        elif k == ord('h'):
            cv.putText(edge_image, 'min' + str(min_val) + '   max' + str(max_val), (30, 30), font, 1, (0, 0, 0),
                       2,
                       cv.LINE_AA)
            lines = cv.HoughLines(edge_image, 1, np.pi / 180, 0)
            cv.imshow('Display', draw_hough_lines(img, lines))
            cv.waitKey(0)
        else:
            print(' o --> increase min threshold \n O --> decrease \n l --> '
                  'decrease max threshold \n L --> increase \n j --> exit')


def draw_hough_lines(img: np.ndarray, lines: np.ndarray, line_number: int = 5) -> np.ndarray:
    """
    Draws the lines defined by rho and theta
    :param line_number: How many line needs to be printed on the image
    :param img: Image to draw on
    :param lines: rho and theta pairs in a matrix
    :return: result image with the lines
    """
    lines = lines[0:line_number]
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img


def draw_general_line(img: np.ndarray, line_params: tuple, color=None) -> np.ndarray:
    """
    Draws a line given with general form (m and b params)
    :param color:Color of the line
    :param img: Image to draw on
    :param line_params: m and b param pair
    :return: result image with the lines
    """
    if color is None:
        color = [0, 0, 255]
    m, b = line_params
    for x in range(0, img.shape[1]):
        y = m * x + b
        y = min(y, img.shape[0])
        y = max(y, 0)
        y = int(y)
        img[y - 1, x, :] = color
    return img
