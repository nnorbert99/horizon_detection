import cv2 as cv
import os
import image_preprocessor as im
import numpy as np

cwd = os.getcwd()
path = os.path.join(cwd, 'Source Images')

image = cv.imread(os.path.join(path, 'example.jpg'))
processed_image = im.preprocess(image)
edge_image = cv.Canny(processed_image, 200, 255, L2gradient=True)
lines = cv.HoughLines(edge_image, 1, np.pi / 180, 0)
processed_image = im.draw_hough_lines(processed_image, lines)
images = [image, processed_image, edge_image]

for img in images:
    cv.imshow('Display', img)
    cv.waitKey(0)
