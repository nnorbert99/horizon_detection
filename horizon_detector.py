import imghdr

import cv2 as cv
import os

from typing import Tuple, List

import image_preprocessor as im
import variance_method as vm
import numpy as np
import argparse

COARSE_SEARCH_HEIGHT = 30
COARSE_SEARCH_WIDTH = 40
FINE_SEARCH_HEIGHT = 300
FINE_SEARCH_WIDTH = 400


def canny_plus_hough_method(pic_paths: [str]) -> None:
    """

    :param pic_paths:
    :return:
    """
    for pic_path in pic_paths:
        image = cv.imread(pic_path)
        processed_image = im.preprocess(image, dsize=(FINE_SEARCH_WIDTH, FINE_SEARCH_HEIGHT))
        edge_image = cv.Canny(processed_image, 200, 255, L2gradient=True)
        lines = cv.HoughLines(edge_image, 1, np.pi / 180, 0)
        processed_image = im.draw_hough_lines(processed_image, lines, 1)
        cv.imshow('Display', processed_image)
        k = cv.waitKey(0)
        if k == ord('f'):
            im.visualise_canny_thresholds(im.preprocess(image))


def variance_method(pic_paths: [str], res_m, res_b) -> List[np.ndarray]:
    """

    :param res_b:
    :param res_m:
    :param pic_paths:
    :return:
    """
    output_pics = []
    for pic_path in pic_paths:
        J = 0
        line = (0, 0)
        image = cv.imread(pic_path)
        processed_image = im.preprocess(image, dsize=(COARSE_SEARCH_WIDTH, COARSE_SEARCH_HEIGHT))
        processed_image = processed_image / 255
        for (current_line) in vm.get_m_and_b(res_m, res_b, (0, COARSE_SEARCH_HEIGHT)):
            current_J = vm.optimization_criterion(processed_image, current_line)
            if current_J > J:
                line = current_line
                J = current_J
        original = im.preprocess(image, dsize=(FINE_SEARCH_WIDTH, FINE_SEARCH_HEIGHT))
        coarse_m, coarse_b = line
        coarse_b = coarse_b * FINE_SEARCH_HEIGHT / COARSE_SEARCH_HEIGHT
        fine_m, fine_b = vm.fine_search(original, (coarse_m, coarse_b), 5)
        im.draw_general_line(original, (coarse_m, coarse_b), color=[255, 255, 255])
        im.draw_general_line(original, (fine_m, fine_b))
        cv.imshow('display', original)
        cv.waitKey(0)
        output_pics.append(original)
    return output_pics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', action='store_true', help='Flag saving the output needed or not')
    parser.add_argument('--output', help='The output folder path')
    parser.add_argument('--input', help='The input folder directory')
    parser.add_argument('--COARSE_SEARCH_RESOLUTION', type=Tuple[int, int], help='Coarse search image resolution')
    parser.add_argument('--FINE_SEARCH_RESOLUTION', type=Tuple[int, int],
                        help='Fine search image resolution also the output resolution')
    parser.add_argument('-r', '--render', action='store_true', help='If rendering the output needed')

    args = parser.parse_args()
    if args.input is None:
        cwd = os.getcwd()
        path = os.path.join(cwd, 'Source Images')
    else:
        path = args.input

    if not os.path.exists(path):
        os.makedirs(path)
        print(f'No Source Image directory found we created one for you at {path}')

    file_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    picture_paths = [f for f in file_paths if imghdr.what(f) in ['jpg', 'png', 'bmp', 'gif', 'tiff', 'jpeg']]

    canny_plus_hough_method(picture_paths)
    variance_method(picture_paths, 10, 1)
