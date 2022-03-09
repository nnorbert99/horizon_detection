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

def visualise_canny_thresholds(img: numpy.ndarray):
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
        else:
            print(' o --> increase min threshold \n O --> decrease \n l --> '
                  'decrease max threshold \n L --> increase \n j --> exit')