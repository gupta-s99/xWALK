"""
Hough Transform File 
adapted from OpenCV documentation for Hough Transform
https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
Last Updated: 22 March 2021

This file computes the Hough Transform on an image of traffic light(s) and uses
the Hough Transform to determine which light (if any) is in the viewer's look
direction. 
"""

import sys
import cv2 as cv
import numpy as np


def redFilter(image):

    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)

    # Range for lower red
    red_lower = np.array([0,120,70])
    red_upper = np.array([10,255,255])
    mask_red1 = cv.inRange(hsv, red_lower, red_upper)

    # Range for upper range
    red_lower = np.array([170,120,70])
    red_upper = np.array([180,255,255])
    mask_red2 = cv.inRange(hsv, red_lower, red_upper)

    mask_red = mask_red1 + mask_red2

    red_output = cv.bitwise_and(image, image, mask=mask_red)
    return red_output

def yellowFilter(image):

    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)

    # Range for Yellow
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    mask_yellow = cv.inRange(hsv, yellow_lower, yellow_upper)

    yellow_output = cv.bitwise_and(image, image, mask=mask_yellow)

    return yellow_output

def greenFilter(image):

    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)

    #Range for Green
    green_lower = np.array([36, 25, 25])
    green_upper = np.array([70, 255,255])
    mask_green = cv.inRange(hsv, green_lower, green_upper)

    green_output = cv.bitwise_and(image, image, mask=mask_green)

    return green_output

def rygFilter(image):

    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)

    # Range for lower red (red is split on hsv scale)
    red_lower = np.array([0,120,70])
    red_upper = np.array([10,255,255])
    mask_red1 = cv.inRange(hsv, red_lower, red_upper)

    # Range for upper range
    red_lower = np.array([170,120,70])
    red_upper = np.array([180,255,255])
    mask_red2 = cv.inRange(hsv, red_lower, red_upper)

    # Range for Yellow
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    mask_yellow = cv.inRange(hsv, yellow_lower, yellow_upper)

    #Range for Green
    green_lower = np.array([36, 25, 25])
    green_upper = np.array([70, 255,255])
    mask_green = cv.inRange(hsv, green_lower, green_upper)

    combined_mask = mask_red1 + mask_red2 + mask_yellow + mask_green

    ryg_output = cv.bitwise_and(image, image, mask=combined_mask)
    return ryg_output


def main(argv):
    ## [load]
    default_file = 'smarties.png'
    filename = argv[0] if len(argv) > 0 else default_file

    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    ## [load]

    ## [Resize Image]
    #want the height to be H = 200 pixels
    H = 200
    scale = src.shape[0] / H
    resized = cv.resize(src, (int(src.shape[1]/scale), int(src.shape[0]/scale)))
    ## [Resize Image]

    ## [Red Filter]
    reddened = redFilter(resized)
    ## [Red Filter]

    ## [Yellow Filter]
    yellowed = yellowFilter(resized)
    ## [Yellow Filter]

    ## [Green Filter]
    greened = greenFilter(resized)
    ## [Green Filter]

    ## [Red, Yellow and Green Filter]
    ryg = rygFilter(resized)
    ## [Red, Yellow, and Green Filter]

    filtered = ryg
    cv.imshow("filtered image", filtered)
    cv.waitKey(0)

    ## [convert_to_gray]
    # Convert it to gray
    gray = cv.cvtColor(filtered, cv.COLOR_BGR2GRAY)
    ## [convert_to_gray]

    ## [reduce_noise]
    # Reduce the noise to avoid false circle detection
    gray = cv.medianBlur(gray, 5)
    ## [reduce_noise]
    
    ## [houghcircles]
    rows = gray.shape[0]
    print(rows)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=20,
                               minRadius=0, maxRadius = 50)
    ## [houghcircles]

    ## [draw]
    print("circles: ", circles)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(resized, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(resized, center, radius, (255, 0, 255), 3)
    ## [draw]

    ## [display]
    cv.imshow("detected circles", resized)
    cv.waitKey(0)
    ## [display]

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])