# import the necessary packages
from __future__ import print_function
import cv2
import numpy as np


image = cv2.imread("green_root.jpg")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lowerBound_yellow = np.array([20,100,100])
upperBound_yellow = np.array([30,255,255])

yellow_mask = cv2.inRange(hsv, lowerBound_yellow, upperBound_yellow)
yellow_mask = cv2.erode(yellow_mask, None, iterations=1)
yellow_mask = cv2.dilate(yellow_mask, None, iterations=1)


im2, yellow_contours, hierarchy = cv2.findContours(yellow_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

((yellow_x, yellow_y), yellow_radius) = cv2.minEnclosingCircle(max(yellow_contours, key=cv2.contourArea))
yellow_center = (int(yellow_x),int(yellow_y))
cm_length = yellow_radius/0.75
cv2.circle(image, yellow_center, int(yellow_radius),(0, 0, 255), 2)
cv2.circle(image, yellow_center, 5, (0, 0, 255), -1)

cv2.imshow("Image", image)

cv2.waitKey(0)
