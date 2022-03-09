import cv2 as cv

import numpy


def preprocess(img: numpy.ndarray, dsize: tuple = (800, 600), mode: str = 'gaussian') -> numpy.ndarray:
    """

    :param img:
    :param dsize:
    :param mode:
    :return:
    """
    image = cv.resize(img, dsize)
    # image = blur_image(image,mode)
    return image


def blur_image(img: numpy.ndarray, mode: str) -> numpy.ndarray:
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