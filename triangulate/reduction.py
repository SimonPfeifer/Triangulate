import numpy as np


def reduction_radial(nodes, sep_min=15):
    check = False
    seperation_min = sep_min
    nodes = np.array(nodes)
    point_first = nodes[0]
    while check == False:
        distance = nodes - nodes[0]
        distance = distance**2
        distance = np.sum(distance, axis=1)
        distance = np.sqrt(distance)
        index = distance > seperation_min
        index[0] = True
        nodes = nodes[index]
        nodes = np.vstack((nodes[1:], nodes[0]))

        if np.all(nodes[0] == point_first):
            check = True

    return nodes.tolist()
