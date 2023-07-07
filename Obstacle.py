import cv2
import numpy as np

# Real-world dimensions of the object (in meters)
object_width = 0.2  # Example width of the object
object_height = 0.2  # Example height of the object

# Focal length (in pixels)
focal_length = 800  # Example focal length

def detect_obstacles(frame):
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the green color range
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Create a binary mask of the green regions
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Perform morphological operations to enhance the mask (if needed)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of the green regions
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area or other criteria
    min_area = 200  # Minimum contour area to consider
    obstacles = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)
            obstacles.append((x, y, w, h))

    return obstacles

def estimate_distance_to_obstacle(observed_width):
    # Estimate the distance to the obstacle using perspective transformation
    estimated_distance = (object_width * focal_length) / observed_width
    return estimated_distance

# Initialize the camera (replace '0' with the appropriate camera index if needed)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect obstacles
    obstacles = detect_obstacles(frame)

    # Check if obstacles are found
    if len(obstacles) > 0:
        # Draw green rectangles around the obstacles
        for (x, y, w, h) in obstacles:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Estimate the distance to the obstacle
            observed_width = w  # Use the width of the bounding rectangle as the observed width
            estimated_distance = estimate_distance_to_obstacle(observed_width)
            print("Obstacle found at a distance of {:.2f} inchs".format(estimated_distance))
        #if estimated_distance <= 0.50:
            #break
    else:
        print("Obstacle not found")

    # Display the resulting frame
    cv2.imshow('Obstacle Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()