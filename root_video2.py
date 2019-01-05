# import the necessary packages
from __future__ import print_function
import cv2
import numpy as np
import time

# Creating array that holds center points
points = []
final_image = 0
# Sets up video capture camera to main camera
cap = cv2.VideoCapture(0)
time_check = time.clock()

while True:
    # load the image and convert it to grayscale
    ret, image = cap.read()
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

    if time.clock() - time_check >= 0.3:
        points.append(center)
        time_check = time.clock()

    for i in range(len(points)-1):
        cv2.line(image, points[i], points[i+1], (0,255,0),5)

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
