import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from scipy.spatial import Delaunay

from detection import corner_detection, edge_detection, frame_detection
from reduction import reduction_radial
from colouration import pixels_inside_triangle, triangle_colour_average

# Load the image and resize it
t0 = time.time()
img = cv2.imread('image.jpg', 0)
img = cv2.resize(img, (640,480))
t1 = time.time()

# Generate nodes from corner and edge detection as well as around the frame of the image 
corners = corner_detection(img)
edges = edge_detection(img)
frame = frame_detection(img)
t2 = time.time()

# Reduce the amount of node based on a radial selection criterion
nodes = frame +corners + edges
nodes = reduction_radial(nodes, sep_min=10)
t3 = time.time()

# Generate Delaunay triangle from the set of nodes
tri = Delaunay(nodes)
t4 = time.time()

# Calculate which pixels are within which triangle
# Calculate the average colour for each set of pixels inside each triangle
# Set every set of pixels for each trianle to its average colour
img_new = np.zeros([np.shape(img)[0], np.shape(img)[1]])
triangle_points = tri.points[tri.vertices]
pixels = [pixels_inside_triangle(nodes) for nodes in triangle_points]
colours = [triangle_colour_average(p, img) for p in pixels]
pixels = np.array([item for sublist in pixels for item in sublist])
colours = [item for sublist in colours for item in sublist]
img_new[pixels[:,1], pixels[:,0]] = colours
t5 = time.time()

# Plot the resulting image
fig1 = plt.figure(1)
plt.imshow(img_new, cmap = 'gray')
plt.xticks([]), plt.yticks([])
t6 = time.time()

# Print basic profiling
print('Reading image: ', t1-t0)
print('Detection: ', t2-t1)
print('Reduction: ', t3-t2)
print('Generating triangles: ', t4-t3)
print('Colouring triangles: ', t5-t4)
print('Plotting final image:', t6-t5)
print('Run time:', t6-t0)

'''
# If enabled, these figures show the step by step process
fig2 = plt.figure(2)
plt.subplot(231)
plt.imshow(img, cmap = 'gray')
plt.title('1. Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(232)
plt.imshow(img, cmap = 'gray')
frame_x, frame_y = list(zip(*frame))
corners_x, corners_y = list(zip(*corners))
edges_x, edges_y = list(zip(*edges))
plt.scatter(x=frame_x, y=frame_y, c='g', s=2)
plt.scatter(x=edges_x, y=edges_y, c='r', s=2)
plt.scatter(x=corners_x, y=corners_y, c='b', s=2)
plt.title('2. Edge and Corner Detection'), plt.xticks([]), plt.yticks([])

plt.subplot(233)
plt.imshow(img, cmap = 'gray')
nodes_x, nodes_y = list(zip(*nodes))
plt.scatter(x=nodes_x, y=nodes_y, c='r', s=2)
plt.title('3. Point Reduction'), plt.xticks([]), plt.yticks([])

plt.subplot(234)
plt.imshow(img, cmap = 'gray')
plt.triplot(nodes_x, nodes_y, tri.simplices.copy(), lw=0.5)
plt.scatter(x=nodes_x, y=nodes_y, c='r', s=2)
plt.title('4. Triangulation'), plt.xticks([]), plt.yticks([])

plt.subplot(235)
plt.imshow(img_new, cmap = 'gray')
plt.title('5. Final Image'), plt.xticks([]), plt.yticks([])
'''
plt.show()
