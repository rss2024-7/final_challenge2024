import math
import cv2 as cv
import numpy as np

def detect_green(img):
    """
    Check if red light detected
	Input:
		src: np.3darray; the input image from camera. BGR.
	Return:
		boolean: True if red light detected, False otherwise
        (x,y): center of image if False, center of light's circle otherwise
        radius: radius of light's circle if detected, 2 otherwise
    """
    h, w, c = img.shape
    
    # Convert BGR to HSV
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV) ## assume H is [0, 180]

    # draw a rectangle to cover top section
    hsv_img = cv.rectangle(hsv_img, (0,0), (w, round(0.2*h)), (0,0,0), -1)
    
    # draw a rectangle to cover right half
    hsv_img = cv.rectangle(hsv_img, (w//2, round(0.2*h)), (w, h), (0,0,0), -1)

    # Blur image before masking
    blur = cv.GaussianBlur(hsv_img, (7,7), 0) #change size of kernel as needed    

    # Mask to keep what we want
    ## Blue HSV, tighten ranges after gaussian blur
    # blue_lower = np.array([50, 150, 230]) # best so far 30, 240, 190
    # blue_upper = np.array([90, 255, 255]) # best for far 30, 255, 255

    # Light Blue/Green HSV
    lb_lower = np.array([82, 102, 207])
    lb_upper = np.array([88, 196, 250])

    ## White HSV
    # white_lower = np.array([0, 0, 240]) # 2 226 107
    # white_upper = np.array([180, 13, 255]) # best: 13, 250, 201

    # mask = cv.inRange(blur, blue_lower, blue_upper)
    # white_mask = cv.inRange(blur, white_lower, white_upper)
    lb_mask = cv.inRange(blur, lb_lower, lb_upper)
    # full_mask = mask | lb_mask

    # Find contours
    contours, hierarchy = cv.findContours(lb_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
      return False, (w//2, h//2), 2
    
    # Find traffic light circle
    max = 0
    # cnt = contours[0]
    cnt = None
    for c in contours:
      x,y,wr,hr = cv.boundingRect(c)
      (x_axis,y_axis), radius = cv.minEnclosingCircle(c)
      area = cv.contourArea(c)
      if area >= max and wr/hr > 0.75 and wr/hr < 1.75 and radius < 15.0:
        max = area
        cnt = c
    if cnt is not None:
      # cv.drawContours(img, contours, -1, (0, 255, 0), 3 )
      (x_axis,y_axis), radius = cv.minEnclosingCircle(cnt)
      # cv.circle(img, (int(x_axis), int(y_axis)), int(radius), (255,0,0), 2)
      # cv2_imshow(img)
      # cv.waitKey()
      return True, (int(x_axis), int(y_axis)), int(radius)

    return False, (w//2, h//2), 2 



def detect_red(img):
    """
    Check if red light detected
	Input:
		src: np.3darray; the input image from camera. BGR.
	Return:
		boolean: True if red light detected, False otherwise
        (x,y): center of image if False, center of light's circle otherwise
        radius: radius of light's circle if detected, 2 otherwise
    """
    h, w, c = img.shape
    
    # Convert BGR to HSV
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV) ## assume H is [0, 180]

    # draw a rectangle to cover top section
    hsv_img = cv.rectangle(hsv_img, (0,0), (w, round(0.2*h)), (0,0,0), -1)
    
    # draw a rectangle to cover right half
    hsv_img = cv.rectangle(hsv_img, (w//2, round(0.2*h)), (w, h), (0,0,0), -1)

    # Blur image before masking
    blur = cv.GaussianBlur(hsv_img, (5,5), 0) #change size of kernel as needed

    # Mask to keep what we want

    ## Red Mask, tighten ranges after gaussian blur
    red_lower = np.array([0, 150, 100]) # best so far 0, 240, 190
    red_upper = np.array([30, 255, 255]) # best for far 30, 255, 255

    ## Yellow Mask
    yellow_lower = np.array([20, 80, 230])
    yellow_upper = np.array([30, 205, 255])

    ## White Mask
    white_lower = np.array([0, 0, 240]) # 2 226 107
    white_upper = np.array([15, 13, 255]) # best: 13, 250, 201

    white_lower2 = np.array([150, 0, 240]) # 2 226 107
    white_upper2 = np.array([180, 13, 255])

    ## Composite Mask
    mask = cv.inRange(blur, red_lower, red_upper)
    white_mask1 = cv.inRange(blur, white_lower, white_upper)
    white_mask2 = cv.inRange(blur, white_lower2, white_upper2)
    yellow_mask = cv.inRange(blur, yellow_lower, yellow_upper)

    full_mask = mask | white_mask1 | white_mask2 | yellow_mask

    # Find contours
    contours, hierarchy = cv.findContours(full_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
      return False, (w//2, h//2), 2
    max = 0
    cnt = None
    for c in contours:
      x,y,wr,hr = cv.boundingRect(c)
      (x_axis,y_axis), radius = cv.minEnclosingCircle(c)
      area = cv.contourArea(c)
      if area >= max and area > 20 and wr/hr > 0.75 and wr/hr < 1.75 and radius < 15.0:
        max = area
        cnt = c

    if cnt is not None:
      (x_axis,y_axis), radius = cv.minEnclosingCircle(cnt)

      return True, (x_axis,y_axis), radius

    return False, (w//2, h//2), 2
