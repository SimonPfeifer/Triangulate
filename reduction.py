import numpy as np


def reduction_radial(points, sep_min=15):
    check = False
    seperation_min = sep_min
    points = np.array(points)
    point_first = points[0]
    while check == False:
        distance = points - points[0]
        distance = distance**2
        distance = np.sum(distance, axis=1)
        distance = np.sqrt(distance)
        index = distance > seperation_min
        index[0] = True
        points = points[index]
        points = np.vstack((points[1:], points[0]))

        if np.all(points[0] == point_first):
            check = True

    return points.tolist()