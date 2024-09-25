import time

import numpy as np
from cv2 import cvtColor, COLOR_RGB2GRAY, COLOR_RGBA2GRAY
from scipy.spatial import Delaunay

from detection import corner_detection, edge_detection, frame_detection
from reduction import reduction_radial
from colour import pixels_inside_triangle, triangle_colour_average

def triangulate(img, corner_threshold, radial_min_sep, profile=False):
    # Check image format and conver to greyscale
    n_colour_channels = 1
    if len(img.shape) == 2:
        img_detection = img
        n_colour_channels = 1
    elif len(img.shape) == 3:
        n_colour_channels = img.shape[2]
        if img.shape[2] == 0:
            img_detection = img
        elif img.shape[2] == 3:
            img_detection = cvtColor(img, COLOR_RGB2GRAY)
        elif img.shape[2] == 4:
            img_detection = cvtColor(img, COLOR_RGBA2GRAY)
        else:
            raise ValueError(f'Unsupported number of colour channels: {img.shape[2]}')
    else:
        raise ValueError(f'Unsupported image format/dimensions: {img.shape}')

    # Generate nodes from corner and edge detection as well as around the frame of the image 
    t1 = time.time()
    corners = corner_detection(img_detection, corner_threshold)
    edges = edge_detection(img_detection)
    frame = frame_detection(img_detection)
    t2 = time.time()

    # Reduce the amount of node based on a radial selection criterion
    nodes = frame +corners + edges
    nodes = reduction_radial(nodes, sep_min=radial_min_sep)
    t3 = time.time()

    # Generate Delaunay triangle from the set of nodes
    tri = Delaunay(nodes)
    t4 = time.time()

    # Calculate which pixels are within which triangle
    # Calculate the average colour for each set of pixels inside each triangle
    # Set every set of pixels for each triangle to its average colour
    img_new = np.zeros([np.shape(img)[0], np.shape(img)[1], n_colour_channels])
    triangle_points = tri.points[tri.simplices]
    pixels = [pixels_inside_triangle(nodes) for nodes in triangle_points]
    colours = [triangle_colour_average(p, img) for p in pixels]
    pixels = np.array([item for sublist in pixels for item in sublist])
    colours = [item for sublist in colours for item in sublist]
    img_new[pixels[:,1], pixels[:,0]] = colours
    t5 = time.time()

    # Print basic profiling
    if profile:
        print(f'Detection: {t2-t1:.4f}s')
        print(f'Reduction: {t3-t2:.4f}s')
        print(f'Generating triangles: {t4-t3:.4f}s')
        print(f'Colouring triangles: {t5-t4:.4f}s')
        print(f'Run time: {t5-t1:.4f}s')

    return img_new
