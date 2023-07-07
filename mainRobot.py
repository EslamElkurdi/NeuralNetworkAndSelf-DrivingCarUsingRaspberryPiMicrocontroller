from MotorModule import Motor
from laneDetect import getLaneCurve
import WebcamModule
import cv2
import JoyStickModule as jsM
from time import sleep

import numpy as np

##################################################
motor = Motor(2, 3, 4, 17, 22, 27)
maxThrottle = 0.8
##################################################
import utlis

def detect_obstacles(frame):
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the green color range
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    #lower_white = np.array([80, 0, 0])
    #upper_white = np.array([255, 160, 255])

    # Create a binary mask of the green regions
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    #White
    #white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)

    # Perform morphological operations to enhance the mask (if needed)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    #White
    #white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    #white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of the green regions
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #white
    #contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area or other criteria
    min_area = 200  # Minimum contour area to consider
    obstacles = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)
            obstacles.append((x, y, w, h))

    '''for contour in contours_white:
        if cv2.contourArea(contour) >= min_a
            obstacles.append((x, y, w, h))'''

    return obstacles
isStop=False
def main():

    """record = 0
    joyVal = jsM.getJS()
    #sleep(0.05)
    print(joyVal['axis1'])
    steering = joyVal['axis1']
    #print(joyVal['axis1'])
    if(joyVal['x']>0):
        throttle = joyVal['x']*maxThrottle
    elif(joyVal['t']>0):
        throttle = joyVal['t']*-maxThrottle
    else:
        throttle = 0
    if joyVal['share'] == 1:
        if record ==0: print('Recording Started ...')
        record +=1
        sleep(0.300)
    if record == 1:
        while record==1 :

            if joyVal['share'] == 1:
                record +=1
                sleep(0.300)
                break;

            img = WebcamModule.getImg(True)
            img = cv2.resize(img, (480, 240))
            curveVal = getLaneCurve(img, 1)

            sen = 1.2  # SENSITIVITY
            maxVAl = 0.8  # MAX SPEED
            if curveVal > maxVAl: curveVal = maxVAl
            if curveVal < -maxVAl: curveVal = -maxVAl
            #print(curveVal)
            if curveVal > 0:
                sen = 1.2
                if curveVal < 0.05: curveVal = 0
            else:
                if curveVal > -0.08: curveVal = 0
            print(curveVal * sen)
            motor.move(0.28, curveVal * sen)
            cv2.waitKey(1)
    elif record == 2:
        record = 0

    else:
        motor.move(throttle,steering)
        cv2.waitKey(1)"""


    img = WebcamModule.getImg(True)
    frame = img
    obstacles = detect_obstacles(frame)
        # Check if obstacles are found
    if len(obstacles) > 0:
        # Draw green rectangles around the obstacles
        for (x, y, w, h) in obstacles:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Estimate the distance to the obstacle
            observed_width = w  # Use the width of the bounding rectangle as the observed width
            #estimated_distance = estimate_distance_to_obstacle(observed_width)
            print("Obstacle found at a distance of inchs")
        #if estimated_distance <= 0.50:
            isStop=True
            motor.stop()
    else:
        print("Obstacle not found")
        isStop=False

    # Display the resulting frame
    cv2.imshow('Obstacle Detection', frame)
    img = cv2.resize(img, (480, 240))
    curveVal = getLaneCurve(img, 1)

    sen = 1.2  # SENSITIVITY
    maxVAl = 0.8  # MAX SPEED
    if curveVal > maxVAl: curveVal = maxVAl
    if curveVal < -maxVAl: curveVal = -maxVAl
    print(curveVal)
    if curveVal > 0:
        sen = 1.2
        if curveVal < 0.05: curveVal = 0
    else:
        if curveVal > -0.08: curveVal = 0
    print(curveVal * sen)
    """if curveVal == 0:
        isStop=True
        motor.stop()"""


    """frame = img
    obstacles = detect_obstacles(frame)
        # Check if obstacles are found
    if len(obstacles) > 0:
        # Draw green rectangles around the obstacles
        for (x, y, w, h) in obstacles:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Estimate the distance to the obstacle
            observed_width = w  # Use the width of the bounding rectangle as the observed width
            #estimated_distance = estimate_distance_to_obstacle(observed_width)
            print("Obstacle found at a distance of inchs")
        #if estimated_distance <= 0.50:
            #break
    else:
        print("Obstacle not found")

    # Display the resulting frame
    cv2.imshow('Obstacle Detection', frame)"""
    if not isStop:
        motor.move(0.28, curveVal * sen)
    cv2.waitKey(1)







if __name__ == '__main__':

    intialTrackBarVals = [102, 80, 20, 214]
    utlis.initializeTrackbars(intialTrackBarVals)

    while True:
        main()