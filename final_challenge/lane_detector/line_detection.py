import cv2 as cv
import numpy as np
import math 
#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv.imshow("image", img)
	cv.waitKey(0)
	cv.destroyAllWindows()

def get_target_pix(src):
	"""
	Find center pixel of detected lane
	Input:
		src: np.3darray; the input image from camera. BGR.
	Return:
		target_pixel: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	########## YOUR CODE STARTS HERE ##########
    h, w, c = src.shape

    # Upper left pix of cropped img
    ul_y = int(3*h/10)
    ul_x = int(w/6)

    # Bottom right pix of cropped img
    br_y = round(.75*h)
    br_x = 5*ul_x

    edge_image = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    ret,thresh1 = cv.threshold(edge_image,150,255,cv.THRESH_BINARY)

    # limit field of view
    crop = thresh1[ul_y:br_y, ul_x:br_x]

    crop = cv.GaussianBlur(crop, (5, 5), 1)

    dst = cv.Canny(crop, 50, 200, None, 3)

    dst = cv.dilate(
        dst,
        cv.getStructuringElement(cv.MORPH_RECT, (5, 5)),
        iterations=1
    )

    dst = cv.erode(
        dst,
        cv.getStructuringElement(cv.MORPH_RECT, (5, 5)),
        iterations=1
    )

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 250, 50)

    slopes = []

    # did not detect any lines
    if linesP is None:
        return None

    # else
    lines = linesP[:][0]
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        p1 = (l[0], l[1])
        p2 = (l[2], l[3])
        slope = None
        y_dist = p2[1] - p1[1]
        x_dist = p2[0] - p1[0]
        if abs(x_dist) != 0:
            slope = (p2[1] - p1[1])/(p2[0]-p1[0])
        else:
            slope = np.inf
        if abs(slope) >= .05 and slope < np.inf:
            slopes.append([i, slope, abs(y_dist)])
    slopes = np.array(slopes)
    
    # sort slopes in ascending order: pos slope is right line, neg slope is left line
    if slopes.shape[0] != 0:
        sorted = slopes[slopes[:, 1].argsort()]
    else:
        return None

    # height, width, channel info of cropped img
    hc, wc, cc = cdstP.shape
    
    # get two steepest lines
    right_lines = sorted[-4:]
    right_lines = right_lines[right_lines[:, 2].argsort()]
    left_lines = sorted[:4]
    left_lines = left_lines[left_lines[:, 2].argsort()]
    l2 = linesP[int(sorted[-1][0])][0]
    l1 = linesP[int(sorted[0][0])][0]

    # if line with max y_dist is geq to lines with max slope and slope of line w/ max y_dist is within 20% of max slope
    if right_lines[-1][2] >= sorted[-1][2] and abs((right_lines[-1][1]-sorted[-1][1])/sorted[-1][1]) <= 0.2:
        l2 = linesP[int(right_lines[-1][0])][0]
    if left_lines[-1][2] > sorted[0][2] and abs((left_lines[-1][1]-sorted[0][1])/sorted[0][1]) <= 0.2:
        l1 = linesP[int(left_lines[-1][0])][0]


    avgx = (l1[2] + l2[0])//2
    avgy = (l1[3] + l2[1])//2
    target_pix = (avgx + ul_x, hc//2 + ul_y)


    # did not detect right line
    if l2[3] <= l2[1]:
        l2 = None
        # use left lane's endpt and add offset to x
        target_pix = ( l1[2] + 30 + ul_x, l1[3] + ul_y)
    
    # did not detect left lane
    if l1[1] <= l1[3]:
        l1 = None
        if l2 is None:
            target_pix = (wc//2 + ul_x, hc//2 + ul_y)
        else:
            target_pix = ( l2[0] - 30 + ul_x, l2[1] + ul_y)

    # further modification: add intersection check? or will that be too far?

    return target_pix, [l1, l2], [ul_x, ul_y]