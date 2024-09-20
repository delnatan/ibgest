from math import dist

from numpy import array, linspace
from scipy.ndimage import map_coordinates


def line_profile(image: array, p1: tuple[int, int], p2: tuple[int, int]):
    """get image intensity profile from two points

    Args:
    image (numpy.ndarray): input image
    p1 (2-tuple of float or int): start coordinate, (row, col).
    p2 (2-tuple of float or int): end coordinate, (row, col).

    Returns:
    1-d numpy.ndarray, interpolated intensity profile
    """

    length = int(dist(p1, p2))
    y1, x1 = p1
    y2, x2 = p2
    xpts = linspace(x1, x2, num=length)
    ypts = linspace(y1, y2, num=length)

    return map_coordinates(image, (ypts, xpts))
