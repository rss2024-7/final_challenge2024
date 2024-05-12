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
    h, w, c = src.shape

    # Upper left pix of cropped img
    ul_y = int(h/3)
    ul_x = int(w/10)

    # Bottom right pix of cropped img
    br_y = 3*ul_y
    br_x = 9*ul_x

    edge_image = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # limit field of view
    crop = edge_image[ul_y:br_y, ul_x:br_x]

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

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 200, 50)

    slopes = []

    # did not detect any lines
    if linesP is None:
        return None

    # else
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        p1 = (l[0], l[1])
        p2 = (l[2], l[3])
        slope = None
        if abs(p2[1] - p1[1]) != 0:
            slope = (p2[1] - p1[1])/(p2[0]-p1[0])
        else:
            slope = np.inf
        slopes.append([i, slope])
    slopes = np.array(slopes)
    
    # sort slopes in ascending order: pos slope is right line, neg slope is left line
    sorted = slopes[slopes[:, 1].argsort()]

    # height, width, channel info of cropped img
    hc, wc, cc = cdstP.shape
    
    # get two steepest lines
    ## l2 is right line
    l2 = linesP[int(sorted[-1][0])][0]
    ## l1 is left line
    l1 = linesP[int(sorted[0][0])][0]
    avgx = (l1[2] + l2[0])//2
    # avgy = (l1[3] + l2[1])//2
    target_pix = (avgx + ul_x, hc//2 + ul_y)

    # p1 = (l1[0] + ul_x , l1[1] + ul_y)
    # p2 = (l1[2] + ul_x , l1[3] + ul_y)
    # p3 = (l2[0] + ul_x , l2[1] + ul_y)
    # p4 = (l2[2] + ul_x , l2[3] + ul_y)

    # did not detect right line
    if l2[3] <= l2[1] or abs(sorted[-1][1]) < 10:
        l2 = None
        # use left lane's endpt and add offset to x
        target_pix = ( l1[2] + 30 + ul_x, l1[3] + ul_y)
    
    if abs(sorted[0][1]) < 5:
         l1 = None
         if l2 is None:
              target_pix = ( w//2, h//2)
         else:
              target_pix = ( l2[0] - 30 + ul_x, l2[1] + ul_y) 

    if l1 is not None:
        p1 = (l1[0] + ul_x , l1[1] + ul_y)
        p2 = (l1[2] + ul_x , l1[3] + ul_y)
    else:
        p1 = None
        p2 = None
    
    if l2 is not None:
        p3 = (l2[0] + ul_x , l2[1] + ul_y)
        p4 = (l2[2] + ul_x , l2[3] + ul_y)
    else:
        p3 = None
        p4 = None    
         

    # further modification: add intersection check? or will that be too far?
    # lines = []
    # p1 = (l1[0] + ul_x , l1[1] + ul_y)
    # p2 = (l1[2] + ul_x , l1[3] + ul_y)
    # p3 = (l2[0] + ul_x , l2[1] + ul_y)
    # p4 = (l2[2] + ul_x , l2[3] + ul_y)


    return target_pix, [p1, p2, p3, p4], [(ul_x, ul_y), (br_x, br_y)], [sorted[-1][0],sorted[0][0]]