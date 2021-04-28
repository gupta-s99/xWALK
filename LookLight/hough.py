"""
Hough Transform File 
adapted from OpenCV documentation for Hough Transform
https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
Last Updated: 28 April 2021

This file computes the Hough Transform on an image of traffic light(s) and uses
the Hough Transform to determine which light (if any) is in the viewer's look
direction. 
"""

import sys
import cv2 as cv
import numpy as np
import time


def redFilter(image, enable=False):

    test = image.copy()
    if enable == True:
        test[:,:,0] = 0
        test[:,:,1] = 0

    hsv = cv.cvtColor(test,cv.COLOR_BGR2HSV)

    # Range for lower red
    red_lower = np.array([0,120,70])
    red_upper = np.array([10,255,255])
    mask_red1 = cv.inRange(hsv, red_lower, red_upper)

    # Range for upper range
    red_lower = np.array([170,120,70])
    red_upper = np.array([180,255,255])
    mask_red2 = cv.inRange(hsv, red_lower, red_upper)

    mask_red = mask_red1 + mask_red2

    red_output = cv.bitwise_and(test, test, mask=mask_red)

    return red_output

def yellowFilter(image):

    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)

    # Range for Yellow
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    mask_yellow = cv.inRange(hsv, yellow_lower, yellow_upper)

    yellow_output = cv.bitwise_and(image, image, mask=mask_yellow)

    return yellow_output

def greenFilter(image, enable=False):

    test = image.copy()
    if enable == True:
        test[:,:,0] = 0
        test[:,:,2] = 0

    hsv = cv.cvtColor(test,cv.COLOR_BGR2HSV)
    # cv.imshow("green hsv image", hsv)
    # cv.waitKey(0)

    #Range for Green
    green_lower = np.array([36, 0, 0])
    green_upper = np.array([85, 255,255])
    mask_green = cv.inRange(hsv, green_lower, green_upper)

    # lower_white = np.array([0,0,0], dtype=np.uint8)
    # upper_white = np.array([0,0,255], dtype=np.uint8)
    # mask_white = cv.inRange(hsv, lower_white, upper_white)

    # mask_comb = mask_green + mask_white

    green_output = cv.bitwise_and(test, test, mask=mask_green)
    
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
    green_lower = np.array([36, 0, 0])
    green_upper = np.array([85, 255,255])
    mask_green = cv.inRange(hsv, green_lower, green_upper)

    combined_mask = mask_red1 + mask_red2 + mask_yellow + mask_green

    ryg_output = cv.bitwise_and(image, image, mask=combined_mask)
    return ryg_output

def rygwFilter(image):

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
    green_lower = np.array([36, 0, 0])
    green_upper = np.array([85, 255,255])
    mask_green = cv.inRange(hsv, green_lower, green_upper)

    #Range for White
    lower_white = np.array([0,0,0], dtype=np.uint8)
    upper_white = np.array([0,0,255], dtype=np.uint8)
    mask_white = cv.inRange(hsv, lower_white, upper_white)

    combined_mask = mask_red1 + mask_red2 + mask_yellow + mask_green + mask_white

    rygw_output = cv.bitwise_and(image, image, mask=combined_mask)
    return rygw_output

# buggy output -> not quite an image...
def histEqual(image):
    # img = cv.imread('fingerptint.jpg', cv.IMREAD_UNCHANGED)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(image)
    cv.imshow("equalized image", dst)
    cv.waitKey(0)
    return dst

# INTENSITY THRESHOLDING ALGORITHM HERE
def intensityThresh(gray, threshold = 0):

    ret, thresh1 = cv.threshold(gray,threshold,255,cv.THRESH_TOZERO)
    return thresh1

def main(argv, color="red"):
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
    H = 300
    scale = src.shape[0] / H
    resized = cv.resize(src, (int(src.shape[1]/scale), int(src.shape[0]/scale)))
    ## [Resize Image]

    ## [Red Filter]
    reddened = redFilter(resized, color == "red")
    ## [Red Filter]

    ## [Yellow Filter]
    yellowed = yellowFilter(resized)
    ## [Yellow Filter]

    ## [Green Filter]
    greened = greenFilter(resized, color == "green")
    ## [Green Filter]

    ## [Red, Yellow and Green Filter]
    ryg = rygFilter(resized)
    ## [Red, Yellow, and Green Filter]

    # ## [Red, Yellow and Green + White Filter]
    # rygw = rygwFilter(resized)
    # ## [Red, Yellow, and Green + White Filter]

    if (color == "red"):
        filtered = reddened
    elif (color == "yellow"):
        filtered = yellowed
    elif (color == "green"):
        filtered = greened
    else:
        filtered = ryg

    

    ## [convert_to_gray]
    # Convert it to gray
    gray = cv.cvtColor(filtered, cv.COLOR_BGR2GRAY)
    ## [convert_to_gray]

    gray = intensityThresh(gray, threshold = 25)
    # cv.imshow("thresholded image", gray)
    # cv.waitKey(0)

    ## [reduce_noise]
    # Reduce the noise to avoid false circle detection
    gray = cv.medianBlur(gray, 5)
    ## [reduce_noise]
    
    ## [houghcircles]
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=20,
                               minRadius=25, maxRadius = 50)
    ## [houghcircles]

    check = gray
    ## [draw]
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(check, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(check, center, radius, (255, 0, 255), 3)
    ## [draw]

    ## [display]
    # cv.imshow("filtered image", filtered)
    # cv.waitKey(0)
    cv.imshow("detected circles", check)
    cv.waitKey(0)
    ## [display]

    return circles


if __name__ == "__main__":
    overall_start = time.time()
    r_circles = main(sys.argv[1:], color="red")
    y_circles = main(sys.argv[1:], color="yellow")
    g_circles = main(sys.argv[1:], color="green")
    num_circles = 0
    color_out = "none"
    if g_circles is not None and (len(g_circles[0]) == 1 or len(g_circles[0]) == 2):
        color_out = "green"
        num_circles = len(g_circles[0])
    if r_circles is not None and len(r_circles[0]) >= 1:
        color_out = "red"
        num_circles = len(r_circles[0])
    if y_circles is not None and len(y_circles[0]) == 1:
        color_out = "yellow"
        num_circles = len(y_circles[0])
    print("total latency:", str(time.time() - overall_start))
    sys.stdout.write(color_out)