# import the necessary packages
from __future__ import print_function
import cv2
import numpy as np


# load the image and convert it to grayscale
image = cv2.imread("root_marker.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lowerBound = np.array([120,50,50])
upperBound = np.array([160,255,255])

mask = cv2.inRange(hsv, lowerBound, upperBound)

im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
c = max(contours, key=cv2.contourArea)
((x, y), radius) = cv2.minEnclosingCircle(c)
center = (int(x),int(y))
cv2.circle(image, (int(x), int(y)), int(radius),(0, 255, 255), 2)
cv2.circle(image, center, 5, (0, 0, 255), -1)

cv2.imshow("Contours", image)


cv2.waitKey(0)
