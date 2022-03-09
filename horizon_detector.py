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
lines = lines[0:5]
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    cv.line(processed_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
images = [image, processed_image, edge_image]

for img in images:
    cv.imshow('Display', img)
    cv.waitKey(0)