import time

import numpy as np
from typing import Tuple

import horizon_detector as hd
import cv2 as cv


def optimization_criterion(img: np.ndarray, line: tuple) -> float:
    """

    :param line:
    :param img:
    :return:
    """
    J = 0
    m,b = line
    ground_pixels = np.ndarray([0, 3])
    sky_pixels = np.ndarray([0, 3])
    for x in range(0, hd.PREPRO_WIDTH):
        y = m*x+b
        y = min(y, hd.PREPRO_HEIGHT)
        y = max(y, 0)
        y = int(y)
        one_collum = img[:, x, :]
        sky_pixels_oc = one_collum[0:y]
        ground_pixels_oc = one_collum[y:hd.PREPRO_HEIGHT]
        ground_pixels = np.concatenate((ground_pixels, ground_pixels_oc))
        sky_pixels = np.concatenate((sky_pixels, sky_pixels_oc))
    if sky_pixels.shape[0] != 1 and ground_pixels.shape[0] != 1:
        ground_covar = np.cov(ground_pixels, rowvar=False)
        sky_covar = np.cov(sky_pixels, rowvar=False)
        if not np.isnan(ground_covar).all() and not np.isnan(sky_covar).all():
            ground_covar_det = np.linalg.det(ground_covar)
            sky_covar_det = np.linalg.det(sky_covar)
            eigenv_ground = np.linalg.eigvals(ground_covar)
            eigenv_sky = np.linalg.eigvals(sky_covar)
            current_J = 1 / (ground_covar_det + sky_covar_det + np.sum(eigenv_ground) ** 2 + np.sum(eigenv_sky) ** 2)
            if current_J > J:
                J = current_J
    return J


def get_theta_r_pairs(resolution_th: int, resolution_r: int) -> Tuple[float, int]:
    """

    :param resolution_th:
    :param resolution_r:
    :return:
    """
    diagonal = int((hd.PREPRO_HEIGHT ** 2 + hd.PREPRO_HEIGHT ** 2) ** (1 / 2))
    for i in range(1, diagonal, resolution_r):
        for k in range(1, 180, resolution_th):
            yield k * (np.pi / 180), i


def get_m_and_b():
    ms = [np.arctan(np.deg2rad(x)) for x in range(-60, 60)]
    for b in range(0, hd.PREPRO_HEIGHT):
        for m in ms:
            yield m, b


if __name__ == '__main__':
    img = np.random.random([hd.PREPRO_HEIGHT, hd.PREPRO_WIDTH, 3])
    for (line) in get_theta_r_pairs(*(10, 1)):
        J_ = optimization_criterion(img, line)
