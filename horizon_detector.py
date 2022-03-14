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


if __name__ == '__main__':
    cwd = os.getcwd()
    path = os.path.join(cwd, 'Source Images')

    if not os.path.exists(path):
        os.makedirs(path)
        print(f'No Source Image directory found we created one for you at {path}')

    file_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    picture_paths = [f for f in file_paths if imghdr.what(f) in ['jpg', 'png', 'bmp', 'gif', 'tiff', 'jpeg']]

    #canny_plus_hough_method(picture_paths)
    for p in picture_paths:
        img = cv.imread(p)
        img = im.preprocess(img)
        vm.optimization_criterion(img,(1,1))
