from typing import Tuple

import numpy as np

import horizon_detector as hd


def optimization_criterion(img: np.ndarray, line: tuple) -> float:
    """

    :param line:
    :param img:
    :return:
    """
    J = 0
    m, b = line
    ground_pixels = np.ndarray([0, 3])
    sky_pixels = np.ndarray([0, 3])
    for x in range(0, img.shape[1]):
        y = m * x + b
        y = min(y, img.shape[0])
        y = max(y, 0)
        y = int(y)
        one_collum = img[:, x, :]
        sky_pixels_oc = one_collum[0:y]
        ground_pixels_oc = one_collum[y:img.shape[0]]
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
    diagonal = int((hd.COARSE_SEARCH_HEIGHT ** 2 + hd.COARSE_SEARCH_HEIGHT ** 2) ** (1 / 2))
    for i in range(1, diagonal, resolution_r):
        for k in range(1, 180, resolution_th):
            yield k * (np.pi / 180), i


def get_m_and_b(m_res: int, b_res: int, b_range_in_pixels: tuple, m_range_in_degree: tuple = (-60, 60)) -> tuple:
    """

    :param b_range_in_pixels:
    :param m_res:
    :param b_res:
    :param m_range_in_degree:
    :return:
    """
    ms = [np.arctan(np.deg2rad(x)) for x in range(*m_range_in_degree, m_res)]
    for b in range(*b_range_in_pixels, b_res):
        for m in ms:
            yield m, b


def fine_search(img: np.ndarray, line_params: tuple, max_iteration_number: int = 50, fine_b_res: int = 10,
                fine_m_res: int = 1) -> tuple:
    """

    :param img:
    :param line_params:
    :param max_iteration_number:
    :return:
    """
    cnt = 0
    best_m, best_b = line_params
    image = img / 255
    J_best = optimization_criterion(image, line_params)
    while True:
        m_decreased = (best_m - np.arctan(np.deg2rad(fine_m_res)), best_b)
        m_increased = (best_m + np.arctan(np.deg2rad(fine_m_res)), best_b)
        b_decreased = (best_m, best_b - fine_b_res)
        b_increased = (best_m, best_b + fine_b_res)
        search_space = {m_decreased: optimization_criterion(image, m_decreased),
                        m_increased: optimization_criterion(image, m_increased),
                        b_decreased: optimization_criterion(image, b_increased),
                        b_increased: optimization_criterion(image, b_decreased)}
        sorted_by_values = dict(sorted(search_space.items(), key=lambda item: item[1], reverse=True))
        best_in_search = list(sorted_by_values.items())[0]
        if best_in_search[1] > J_best:
            J_best = best_in_search[1]
            best_m, best_b = best_in_search[0]
            was_better = True
        else:
            was_better = False
        cnt += 1
        print(cnt)
        if cnt == max_iteration_number or not was_better:
            break
    return best_m, best_b


if __name__ == '__main__':
    img = np.random.random([hd.COARSE_SEARCH_HEIGHT, hd.COARSE_SEARCH_WIDTH, 3])
    for (line) in get_theta_r_pairs(*(10, 1)):
        J_ = optimization_criterion(img, line)
