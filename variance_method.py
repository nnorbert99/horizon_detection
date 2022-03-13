import time

import numpy as np
from typing import Tuple

import horizon_detector as hd
import cv2 as cv


def optimization_criterion(img: np.ndarray, line: tuple) -> float:
    pass


def is_above_line(point: tuple, line: tuple) -> bool:
    pass


def get_theta_r_pairs(resolution_th: int, resolution_r: int) -> Tuple[float, int]:
    diagonal = int((hd.PREPRO_HEIGHT ** 2 + hd.PREPRO_HEIGHT ** 2) ** (1 / 2))
    for i in range(0, diagonal, resolution_r):
        for k in range(1, 180, resolution_th):
            yield k * (np.pi / 180), i


if __name__ == '__main__':
    start = time.time()
    img = np.zeros([hd.PREPRO_HEIGHT, hd.PREPRO_WIDTH, 3])
    for th, r in get_theta_r_pairs(10, 10):
        for x in range(0, hd.PREPRO_WIDTH):
            y = (r - x * np.cos(th)) / np.sin(th)
            if 0 < y < hd.PREPRO_HEIGHT - 1:
                img[int(y), x] = [255, 255, 255]
    end = time.time()
    print(str((end-start)*1000) +'ms')
    cv.imshow('Display', img)
    cv.waitKey(0)
