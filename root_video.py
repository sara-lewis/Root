# import the necessary packages
from __future__ import print_function
import cv2
import numpy as np
import time

# Creating array that holds center points
points = []
final_image = 0
time_check = time.clock()

# Sets up video capture camera to main camera
cap = cv2.VideoCapture(0)

while True:
# Reads the video capture to get the frame
    ret, image = cap.read()
# Changes image to hue/saturation/value format
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Define bounds for color I want to track
    lowerBound = np.array([50,50,50])
    upperBound = np.array([70,255,255])
# Create mask and get rid of noise
    mask = cv2.inRange(hsv, lowerBound, upperBound)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

# Create contours from mask and draw them
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#    cv2.drawContours(image, contours, -1, (0,0,255), 3)

# Average contour points to get the center point
    if len(contours)>0:
        x_val = 0
        y_val = 0
        for i in range(len(contours)):
            x_val += contours[i][0]
            y_val += contours[i][1]

        x_val = x_val/len(contours)
        y_val = y_val/len(contours)
# Create center array for later steps
        cnt = np.array([x_val, y_val])
# Make small circle on screen
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        cv2.circle(image,center,5,(0,255,0),2)

        if time.clock() - time_check >= 0.3:
            points.append(center)
            time_check = time.clock()

    for i in range(len(points)-1):
        cv2.line(image, points[i], points[i+1], (0,255,0),5)

# Show current frame
    cv2.imshow("Video", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        final_image = image
        break

# When everything done, release the capture
cap.release()
cv2.destroyWindow("Video")
# Show final picture with tracking
cv2.imshow("Final image", final_image)
cv2.waitKey(0)
