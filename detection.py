import cv2
import numpy as np

def corner_detection(img, th=25):
    fast = cv2.FastFeatureDetector_create(threshold=th)
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

    print('Corner Detction Threshold: ', fast.getThreshold())
    print('Total points: ', len(kp))

    points = [key.pt for key in kp]

    return points


def edge_detection(img):
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    threshold_min = 0.66*avg_color 
    threshold_max = 1.33*avg_color
    edges = cv2.Canny(img, threshold_min, threshold_max)

    x, y = np.nonzero(edges)
    points = list(zip(y,x))
    
    print('Edge Detction Threshold: ', 'Min', int(threshold_min), '  Max', int(threshold_max))
    print('Total points: ', len(points))

    return points
