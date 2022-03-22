import cv2 as cv

import numpy as np


def preprocess(img: np.ndarray, dsize: tuple = (800, 600), mode: str = 'gaussian') -> np.ndarray:
    """

    :param img:
    :param dsize:
    :param mode:
    :return:
    """
    image = cv.resize(img, dsize)
    # image = blur_image(image,mode)
    return image


def blur_image(img: np.ndarray, mode: str) -> np.ndarray:
    """

    :param img:
    :param mode:
    :return:
    """
    if mode == 'gaussian':
        sigma = img.shape[1] / 50
        return cv.GaussianBlur(img, (5, 5), sigma)
    else:
        print('Invalid mode')
        return None


def visualise_canny_thresholds(img: np.ndarray):
    """

    :param img:
    :return:
    """
    min_val = 0
    max_val = 255
    font = cv.FONT_HERSHEY_SIMPLEX

    while True:
        edge_image = cv.Canny(img, min_val, max_val, L2gradient=True)
        cv.putText(edge_image, 'min' + str(min_val) + '   max' + str(max_val), (30, 30), font, 1, (255, 255, 255), 2,
                   cv.LINE_AA)
        cv.imshow('Display', edge_image)
        k = cv.waitKey(0)
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

    :param line_number:
    :param img:
    :param lines:
    :return:
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


def draw_line(img: np.ndarray, line: tuple) -> np.ndarray:
    """

    :param img:
    :param line:
    :return:
    """
    rho, theta = line
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
