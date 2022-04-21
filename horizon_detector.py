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


def canny_plus_hough_method(pic_paths: [str], render: bool = True) -> List[Tuple[np.ndarray, str]]:
    """
    Runs the canny edge detector with the hough transform to identify the most prominent line in the picture
    :param render: if render is required needs to be set
    :param pic_paths: Path to the pictures
    :return:
    """
    output_pics = []
    # Iterate through the pictures
    for pic_path in pic_paths:
        # Reading images into np.ndarrays
        image = cv.imread(pic_path)
        # Resize the image
        processed_image = im.preprocess(image, dsize=(FINE_SEARCH_WIDTH, FINE_SEARCH_HEIGHT))
        # Run the Canny edge detector gives back an edge map
        edge_image = cv.Canny(processed_image, 200, 255, L2gradient=True)
        # In the binary map search for potencial lines
        lines = cv.HoughLines(edge_image, 1, np.pi / 180, 0)
        # Draw the line into the image
        hough_image = im.draw_hough_lines(processed_image, lines, 1)
        if render:
            # If render display the image and wait for input
            cv.imshow('Display', hough_image)
            k = cv.waitKey(0)
            if k == ord('f'):
                # By pressing f thresholds can be seen
                im.visualise_canny_thresholds(im.preprocess(image))
        filename = os.path.basename(pic_path)
        # return with the final image with the name of that image
        output_pics.append((hough_image, filename))
    return output_pics


def variance_method(pic_paths: [str], res_m, res_b, render=True) -> List[Tuple[np.ndarray, str]]:
    """
    Runs the variance method on the images
    :param render: if render needs to be set
    :param res_b: resolution for the b parameter in the line equation
    :param res_m: resolution for the m parameter in the line equation
    :param pic_paths: path to the pictures
    :return:
    """
    output_pics = []
    for pic_path in pic_paths:
        J = 0
        line = (0, 0)
        image = cv.imread(pic_path)
        processed_image = im.preprocess(image, dsize=(COARSE_SEARCH_WIDTH, COARSE_SEARCH_HEIGHT))
        # Normalize the intensity values for better J values
        processed_image = processed_image / 255
        # Iterate through the possible m and b combinations
        for (current_line) in vm.get_m_and_b(res_m, res_b, (0, COARSE_SEARCH_HEIGHT)):
            # calculate the optimization criterion for that line
            current_J = vm.optimization_criterion(processed_image, current_line)
            # If that line fits better keep it (maximum search)
            if current_J > J:
                line = current_line
                J = current_J
        # Get the original image reshaped to the fine search resolution
        original = im.preprocess(image, dsize=(FINE_SEARCH_WIDTH, FINE_SEARCH_HEIGHT))
        coarse_m, coarse_b = line
        # Scale the b parameter to the bigger resolution
        coarse_b = coarse_b * FINE_SEARCH_HEIGHT / COARSE_SEARCH_HEIGHT
        # Run the fine search for a better line
        fine_m, fine_b = vm.fine_search(original, (coarse_m, coarse_b), 5)
        # Draw the coarse line with white color
        im.draw_general_line(original, (coarse_m, coarse_b), color=[255, 255, 255])
        # Draw the fine line with red color
        im.draw_general_line(original, (fine_m, fine_b))
        if render:
            cv.imshow('display', original)
            cv.waitKey(0)
        filename = os.path.basename(pic_path)
        output_pics.append((original, filename))
    return output_pics


if __name__ == '__main__':
    # Argument parsing feature for the tunable parameters and flags
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', action='store_true', help='Flag saving the output needed or not')
    parser.add_argument('--output', help='The output folder path')
    parser.add_argument('--input', help='The input folder directory')
    parser.add_argument('-csr', '--COARSE_SEARCH_RESOLUTION', type=int, nargs='+',
                        help='Coarse search image resolution')
    parser.add_argument('-fsr', '--FINE_SEARCH_RESOLUTION', type=int, nargs='+',
                        help='Fine search image resolution also the output resolution')
    parser.add_argument('-nr', '--no_render', action='store_false', help='If rendering the output is not needed')

    args = parser.parse_args()
    if args.COARSE_SEARCH_RESOLUTION is not None:
        # Change the res if it has given
        COARSE_SEARCH_WIDTH, COARSE_SEARCH_HEIGHT = args.COARSE_SEARCH_RESOLUTION
    if args.FINE_SEARCH_RESOLUTION is not None:
        # Change the res if it has given
        FINE_SEARCH_WIDTH, FINE_SEARCH_HEIGHT = args.FINE_SEARCH_RESOLUTION
    if args.input is None:
        # Change the source folder path if given
        cwd = os.getcwd()
        path = os.path.join(cwd, 'Source Images')
    else:
        path = args.input

    if not os.path.exists(path):
        os.makedirs(path)
        print(f'No Source Image directory found we created one for you at {path}')
    # Get every file in the path
    file_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # Search in these paths for images
    picture_paths = [f for f in file_paths if imghdr.what(f) in ['jpg', 'png', 'bmp', 'gif', 'tiff', 'jpeg']]

    # Run the methods and get the results
    can_output = canny_plus_hough_method(picture_paths, render=args.no_render)
    var_output = variance_method(picture_paths, 10, 1, render=args.no_render)
    # If Save is needed save the results
    if args.save:
        if args.output is None:
            # If output folder is given save it to that otherwise create a folder
            cwd = os.getcwd()
            output_path = os.path.join(cwd, 'Output Images')
        else:
            output_path = args.output
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # Saving pictures
        for pic, filename in var_output:
            output_full_path = os.path.join(output_path, 'variance_' + filename)
            cv.imwrite(output_full_path, pic)
        for pic, filename in can_output:
            output_full_path = os.path.join(output_path, 'canny_' + filename)
            cv.imwrite(output_full_path, pic)
        print(f'Saving pictures to {output_path}')