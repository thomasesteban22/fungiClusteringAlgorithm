import numpy as np
#Script Best Distance Function

def manhattan_distance(point1, point2):
    """
    Calculate the Manhattan distance between two points.

    The Manhattan distance, also known as L1 distance, is the sum of the absolute
    differences between the coordinates of two points.

    Parameters
    ----------
    point1 : ndarray
        The first point as a NumPy array.
    point2 : ndarray
        The second point as a NumPy array.

    Returns
    -------
    float
        The Manhattan distance between point1 and point2.
    """
    return np.sum(np.abs(point1 - point2))
