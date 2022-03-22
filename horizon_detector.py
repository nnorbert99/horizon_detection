import imghdr

import cv2 as cv
import os
import image_preprocessor as im
import variance_method as vm
import numpy as np

PREPRO_HEIGHT = 60
PREPRO_WIDTH = 80


def canny_plus_hough_method(pic_paths: [str]) -> None:
    """

    :param pic_paths:
    :return:
    """
    for pic_path in picture_paths:
        image = cv.imread(pic_path)
        processed_image = im.preprocess(image, dsize=(PREPRO_WIDTH, PREPRO_HEIGHT))
        edge_image = cv.Canny(processed_image, 200, 255, L2gradient=True)
        lines = cv.HoughLines(edge_image, 1, np.pi / 180, 0)
        processed_image = im.draw_hough_lines(processed_image, lines, 3)
        cv.imshow('Display', processed_image)
        k = cv.waitKey(0)
        if k == ord('f'):
            im.visualise_canny_thresholds(im.preprocess(image))


def variance_method(pic_paths: [str], res_th, res_r) -> None:
    """

    :param pic_paths:
    :param res_th:
    :param res_r:
    :return:
    """
    for pic_path in pic_paths:
        J = 0
        line = (0, 0)
        image = cv.imread(pic_path)
        processed_image = im.preprocess(image, dsize=(PREPRO_WIDTH, PREPRO_HEIGHT))
        processed_image = processed_image/255
        for (current_line) in vm.get_m_and_b():
            current_J = vm.optimization_criterion(processed_image, current_line)
            if current_J > J:
                line = current_line
                J = current_J
        original = im.preprocess(image, dsize=(PREPRO_WIDTH, PREPRO_HEIGHT))
        m,b = line
        for x in range(0,PREPRO_WIDTH):
            y = m*x+b
            y = min(y, PREPRO_HEIGHT)
            y = max(y, 0)
            y = int(y)
            original[y-1,x,:] = [0,0,255]
        cv.imshow('display',original)
        cv.waitKey(0)

if __name__ == '__main__':
    cwd = os.getcwd()
    path = os.path.join(cwd, 'Source Images')

    if not os.path.exists(path):
        os.makedirs(path)
        print(f'No Source Image directory found we created one for you at {path}')

    file_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    picture_paths = [f for f in file_paths if imghdr.what(f) in ['jpg', 'png', 'bmp', 'gif', 'tiff', 'jpeg']]

    # canny_plus_hough_method(picture_paths)
    variance_method(picture_paths,10,1)
