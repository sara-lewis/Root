# import the necessary packages
from __future__ import print_function
import cv2
import numpy as np


# load the image and convert it to grayscale
image = cv2.imread("green_root4.jpg")
middle_image = image.copy()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Placeholder numbers for now -- use digital color meter
lowerBound = np.array([30,50,50])
upperBound = np.array([90,255,255])

mask = cv2.inRange(hsv, lowerBound, upperBound)
mask = cv2.erode(mask, None, iterations=1)
mask = cv2.dilate(mask, None, iterations=1)


im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(middle_image, contours, -1, (0,0,255), 20)

mask = cv2.inRange(middle_image, (0,0,250), (0,0,255))
mask = cv2.dilate(mask, None, iterations=5)
im1, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

c = max(contours, key=cv2.contourArea)
((x, y), radius) = cv2.minEnclosingCircle(c)
center = (int(x),int(y))
cv2.circle(image, (int(x), int(y)), int(radius),(0, 255, 0), 2)
cv2.circle(image, center, 5, (0, 255, 0), -1)


cv2.imshow("Middle Image", middle_image)
cv2.imshow("Image", image)



cv2.waitKey(0)
