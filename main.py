import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial import Delaunay

from detection import corner_detection, edge_detection, frame_detection
from reduction import reduction_radial
from colouration import points_inside_triangle, triangle_colour_average, triangle_colourise


img = cv2.imread('gandalf.jpg', 0)
img = cv2.resize(img, (640,480))

corners = corner_detection(img)
edges = edge_detection(img)
frame = frame_detection(img)

'''
plt.subplot(131),plt.imshow(img, cmap = 'gray')
corners_x , corners_y = list(zip(*corners))
edges_x , edges_y = list(zip(*edges))
plt.scatter(x=edges_x, y=edges_y, c='r', s=2)
plt.scatter(x=corners_x, y=corners_y, c='b', s=2)
plt.title('Edge and Corner Detection'), plt.xticks([]), plt.yticks([])

plt.subplot(132),plt.imshow(img, cmap = 'gray')
points = corners + edges
points_new = reduction_radial(points, sep_min=10)
points_x , points_y = list(zip(*points_new))
plt.scatter(x=points_x, y=points_y, c='r', s=2)
plt.title('Point Reduction'), plt.xticks([]), plt.yticks([])

plt.subplot(133),plt.imshow(img, cmap = 'gray')
tri = Delaunay(points_new)
plt.imshow(img, cmap = 'gray')
plt.triplot(points_x, points_y, tri.simplices.copy(), lw=0.5)
plt.scatter(x=points_x, y=points_y, c='r', s=2)
plt.title('Triangulation'), plt.xticks([]), plt.yticks([])

plt.show()
'''

points = corners + edges + frame
points_new = reduction_radial(points, sep_min=10)
points_x , points_y = list(zip(*points_new))
tri = Delaunay(points_new)

plt.subplot(131)
plt.imshow(img, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(132)
plt.imshow(img, cmap = 'gray')
plt.triplot(points_x, points_y, tri.simplices.copy(), lw=0.5)
plt.scatter(x=points_x, y=points_y, c='r', s=2)
plt.title('Nodes and Triangles'), plt.xticks([]), plt.yticks([])

img_new = np.zeros([np.shape(img)[0], np.shape(img)[1]])
total = 0
for i in tri.vertices:
    A = tri.points[i[0]]
    B = tri.points[i[1]]
    C = tri.points[i[2]]

    points = points_inside_triangle(A, B, C)
    value = triangle_colour_average(points, img)
    img_new = triangle_colourise(points, img_new, value)


plt.subplot(133)
plt.imshow(img_new, cmap = 'gray')
plt.title('Triangulation'), plt.xticks([]), plt.yticks([])
plt.show()