"""
-This module gets an image through the webcam
using the opencv package
-Display can be turned on or off
-Image size can be defined
"""

import cv2

cap = cv2.VideoCapture(0)

def getImg(display= False,size=[480,240]):
    _, img = cap.read()
    # img = cv2.imread(imgg)
    if img is not None:
        # Specify the desired output size
        desired_size = (480, 240)  # Replace with your desired width and height

        # Resize the image
        img = cv2.resize(img, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)

        print('load the image.')

        # Do further processing with the resized image
    else:
        print('Failed to load the image.')

    # ...................................................
    if display:
        cv2.imshow('IMG',img)
        cv2.waitKey(1)
    return img

if __name__ == '__main__':
    while True:
        img = getImg(True)