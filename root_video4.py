# import the necessary packages
import cv2
import numpy as np
import math
import time

# Initialize variables that will be used in the program
points = []
final_image = 0
raw_vals = []
frames = 0
start_time = time.time()
end_time = 0
y, x, radius = 0, 0, 0
minX, minY = 0, 0

# Sets up video capture camera to main camera
cap = cv2.VideoCapture(0)

while True:
    frames += 1
    # Load the current video frame
    ret, image = cap.read()
    video_frame = image.copy()
    # Crop frame if results have already indicated robot's general location
    if radius > 0 and (x-2*radius) > 0 and (y-2*radius) > 0:
        minX, minY = int(x-2*radius), int(y-2*radius)
        image = image[minY:int(y+2*radius), minX:int(x+2*radius)]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    middle_image = image.copy()

    # Set boundries for green light
    lowerBound = np.array([30,50,50], dtype=np.uint8)
    upperBound = np.array([90,255,255], dtype=np.uint8)

    # Make a mask with the boundries and get rid of noise
    mask = cv2.inRange(hsv, lowerBound, upperBound)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    # Find contours from first mask
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # If contours exist the rest of this will be done
    if len(contours)>0:
        # Draw contours on copy image so they can be further analyzed
        cv2.drawContours(middle_image, contours, -1, (0,0,255), 20)

        # Make a new mask from the previous red contours to get more data
        mask = cv2.inRange(middle_image, (0,0,250), (0,0,255))
        mask = cv2.dilate(mask, None, iterations=5)
        im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # Make circle around all contours and find center point
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        # Orient the center to the uncropped image
        x += minX
        y += minY
        center = (int(x),int(y))
#        cv2.circle(video_frame, (int(x), int(y)), int(radius),(0, 0, 255), 2)
        cv2.circle(video_frame, center, 5, (0, 255, 0), -1)
        center = [int(x), int(y)]


        # Average past couple points and update array
        if len(raw_vals) == 4:
            average_x = 0
            average_y = 0
            for i in range(len(raw_vals)):
                average_x += raw_vals[i][0]
                average_y += raw_vals[i][1]
            average_x = average_x/len(raw_vals)
            average_y = average_y/len(raw_vals)
            average_val = (average_x, average_y)
            points.append(average_val)
            raw_vals = []
        else:
            raw_vals.append(center)
    else:
        x, y, radius = 0, 0, 0

    # Print the lines on the screen that represent previous movement
    for i in range(len(points)-1):
        cv2.line(video_frame, points[i], points[i+1], (0,255,0),5)

    # Rescale image so it fits on computer screen
    r = 1400.0 / video_frame.shape[1]
    dim = (1400, int(video_frame.shape[0] * r))
    video_frame = cv2.resize(video_frame, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("Video", video_frame)

    # Quit video if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        final_image = video_frame.copy()
        end_time = time.time()
        break

# When everything done, release the capture
cap.release()
cv2.destroyWindow("Video")

# Show final picture with tracking
fps = str(int(frames/(end_time - start_time)))
cv2.putText(final_image,fps + " FPS",(10,500), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2,cv2.LINE_AA)
cv2.imshow("Final image", final_image)
cv2.waitKey(0)
