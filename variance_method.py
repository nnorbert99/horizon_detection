from typing import Tuple

import numpy as np

import horizon_detector as hd


def optimization_criterion(img: np.ndarray, line: tuple) -> float:
    """
    Calculate the optimization criterion value for a line and an image.
    :param line: m and b params that define the line
    :param img: image as an np.ndarray
    :return: J value
    """
    # initialize the values
    J = 0
    m, b = line
    ground_pixels = np.ndarray([0, 3])
    sky_pixels = np.ndarray([0, 3])
    # iterate from 0 to width
    for x in range(0, img.shape[1]):
        # equation of the line
        y = m * x + b
        # cut the values that is higher or lower that the height of the image
        y = min(y, img.shape[0])
        y = max(y, 0)
        # convert it to an integer number
        y = int(y)
        # cut the x-th collum from the picture
        one_collum = img[:, x, :]
        # separate the collum into 2 section (ground and sky) based on the line (below and above)
        sky_pixels_oc = one_collum[0:y]
        ground_pixels_oc = one_collum[y:img.shape[0]]
        # concatenate these collum parts into one matrix
        ground_pixels = np.concatenate((ground_pixels, ground_pixels_oc))
        sky_pixels = np.concatenate((sky_pixels, sky_pixels_oc))
    # if neither of the matrices are empty
    if sky_pixels.shape[0] != 1 and ground_pixels.shape[0] != 1:
        # calculate covariance matrix for ground and sky pixels
        ground_covar = np.cov(ground_pixels, rowvar=False)
        sky_covar = np.cov(sky_pixels, rowvar=False)
        # if data is valid
        if not np.isnan(ground_covar).all() and not np.isnan(sky_covar).all():
            # calculate the determinant of the covariance matrices
            ground_covar_det = np.linalg.det(ground_covar)
            sky_covar_det = np.linalg.det(sky_covar)
            # calculate the eigenvalues of the covariance matrices
            eigenv_ground = np.linalg.eigvals(ground_covar)
            eigenv_sky = np.linalg.eigvals(sky_covar)
            # calculate the J values
            current_J = 1 / (ground_covar_det + sky_covar_det + np.sum(eigenv_ground) ** 2 + np.sum(eigenv_sky) ** 2)
            # If there was valid data return with it otherwise 0
            if current_J > J:
                J = current_J
    return J


def get_theta_r_pairs(resolution_th: int, resolution_r: int) -> Tuple[float, int]:
    """
    Generate theta and r pairs for hough line representation (not used after all)
    :param resolution_th: Resolution for the theta parameter
    :param resolution_r: Resolution for the rho parameter
    :return: theta rho pair
    """
    diagonal = int((hd.COARSE_SEARCH_HEIGHT ** 2 + hd.COARSE_SEARCH_HEIGHT ** 2) ** (1 / 2))
    for i in range(1, diagonal, resolution_r):
        for k in range(1, 180, resolution_th):
            yield k * (np.pi / 180), i


def get_m_and_b(m_res: int, b_res: int, b_range_in_pixels: tuple, m_range_in_degree: tuple = (-60, 60)) -> tuple:
    """
    Generate m and b parameter for general line representation
    :param b_range_in_pixels: b param's range in pixels(image height)
    :param m_res: Resolution for m parameter degree
    :param b_res: Resolution for b parameter pixel
    :param m_range_in_degree: m parameter's range tuple in degrees
    :return:m and b param pair
    """
    # Generate the slope (m)  values based on the degree of the line and x axis
    ms = [np.arctan(np.deg2rad(x)) for x in range(*m_range_in_degree, m_res)]
    # Go down (height) the pictures with the b values
    for b in range(*b_range_in_pixels, b_res):
        for m in ms:
            yield m, b


def fine_search(img: np.ndarray, line_params: tuple, max_iteration_number: int = 50, fine_b_res: int = 10,
                fine_m_res: int = 1) -> tuple:
    """
    Performs a fine search based on the coarse result
    :param fine_m_res: Resolution of the m parameter at the fine search degree
    :param fine_b_res: Resolution of the b parameter at the fine search pixel
    :param img: Image in np.ndarray format
    :param line_params: the found m and b params at the coarse search
    :param max_iteration_number: when to stop the search
    :return: The result m and b pair of the fine search
    """
    # initialize
    cnt = 0
    best_m, best_b = line_params
    # normalize the image
    image = img / 255
    # calculate the J value for the coarse search's result
    J_best = optimization_criterion(image, line_params)
    while True:
        # Move in the parameter space in 4 directions b+,b-,m+,m-
        m_decreased = (best_m - np.arctan(np.deg2rad(fine_m_res)), best_b)
        m_increased = (best_m + np.arctan(np.deg2rad(fine_m_res)), best_b)
        b_decreased = (best_m, best_b - fine_b_res)
        b_increased = (best_m, best_b + fine_b_res)
        # calculate J values for each small changed line
        search_space = {m_decreased: optimization_criterion(image, m_decreased),
                        m_increased: optimization_criterion(image, m_increased),
                        b_decreased: optimization_criterion(image, b_increased),
                        b_increased: optimization_criterion(image, b_decreased)}
        # sort the dict by values in descending order so the line with the highest J value is first
        sorted_by_values = dict(sorted(search_space.items(), key=lambda item: item[1], reverse=True))
        # get the best line (defined as a param pair) and the corresponding J value
        best_in_search = list(sorted_by_values.items())[0]
        # If there is a better line in the parameter space than the coarse result move in that direction
        if best_in_search[1] > J_best:
            # update the best J value
            J_best = best_in_search[1]
            # update the line params
            best_m, best_b = best_in_search[0]
            was_better = True
        else:
            was_better = False
        cnt += 1
        print(cnt)
        # break when no more moves left or no improvement reached
        if cnt == max_iteration_number or not was_better:
            break
    return best_m, best_b


if __name__ == '__main__':
    img = np.random.random([hd.COARSE_SEARCH_HEIGHT, hd.COARSE_SEARCH_WIDTH, 3])
    for (line) in get_theta_r_pairs(*(10, 1)):
        J_ = optimization_criterion(img, line)
