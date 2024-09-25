import numpy as np
from cv2 import FastFeatureDetector_create, Canny


def corner_detection(img, threshold=25):
    fast = FastFeatureDetector_create(threshold=int(threshold))
    kp = fast.detect(img, None)

    points = [int(value) for key in kp for value in key.pt]
    points = list(zip(points[::2], points[1::2]))

    return points

def edge_detection(img):
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    threshold_min = 0.66*avg_color 
    threshold_max = 1.33*avg_color
    edges = Canny(img, threshold_min, threshold_max)

    x, y = np.nonzero(edges)
    points = list(zip(y,x))

    return points

def frame_detection(img, point_spacing = 100, n_nodes_vert=None, n_nodes_hori=None):
    img_vert = np.shape(img)[0]
    img_hori = np.shape(img)[1]

    if n_nodes_vert == None:
        n_nodes_vert = int(img_vert / point_spacing)

    if n_nodes_hori == None:
        n_nodes_hori = int(img_hori / point_spacing)

    range_vert = np.linspace(0, img_vert-1, num=n_nodes_vert, dtype=int)
    range_hori = np.linspace(0, img_hori-1, num=n_nodes_hori, dtype=int)

    left = list(zip(np.zeros(n_nodes_vert), range_vert))
    right = list(zip([img_hori-1]*n_nodes_vert, range_vert))
    bottom = list(zip(range_hori, np.zeros(n_nodes_hori)))
    top = list(zip(range_hori, [img_vert-1]*n_nodes_hori))

    return left + right + top + bottom
