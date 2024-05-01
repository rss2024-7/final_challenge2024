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
    ul_y = int(h/4)
    ul_x = int(w/6)

    # Bottom right pix of cropped img
    br_y = 3*ul_y
    br_x = 5*ul_x

    edge_image = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    crop = edge_image[ul_y:br_y, ul_x:br_x]
    # cv2_imshow(crop)

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


    # lines = cv.HoughLines(dst, 1, np.pi / 180, 150, 80, 0, 0)

    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)


    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 200, 50)

    slopes = []

    if linesP is None:
        return None

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            p1 = (l[0], l[1])
            p2 = (l[2], l[3])
            slope = None
            if abs(p2[1] - p1[1]) != 0:
              slope = (p2[1] - p1[1])/(p2[0]-p1[0])
              # if abs(slope) > 0.2:
              #   cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
            else:
              slope = np.inf
            slopes.append([i, abs(slope)])
    slopes = np.array(slopes)
    # sorted = np.flip(np.sort(slopes))  # sorts slopes in descending order
    sorted = slopes[slopes[:, 1].argsort()]

    # height, width, channel info of cropped img
    hc, wc, cc = cdstP.shape
    if linesP is not None:
        # get two steepest lines
        l1 = linesP[int(sorted[-1][0])][0]
        l2 = linesP[int(sorted[-2][0])][0]
        cv.line(cdstP, (l1[0], l1[1]), (l1[2], l1[3]), (0,0,255), 3, cv.LINE_AA)
        cv.line(cdstP, (l2[0], l2[1]), (l2[2], l2[3]), (0,0,255), 3, cv.LINE_AA)
        avgx = (l1[2] + l2[0])//2
        # avgy = (l1[3] + l2[1])//2

        # draw center pixel on cropped img: using half of cropped height as lookahead dist
        cv.circle(cdstP, (avgx, hc//2), radius=4, color=(255, 0, 0), thickness=-1)
        
        # draw detected lines + center pixel on src img: add offset from cropping
        cv.circle(src, (avgx + ul_x, hc//2 + ul_y), radius=4, color=(255, 0, 0), thickness=-1)
        cv.line(src, (l1[0] + ul_x, l1[1] + ul_y), (l1[2] + ul_x, l1[3] + ul_y), (0,0,255), 3, cv.LINE_AA)
        cv.line(src, (l2[0] + ul_x, l2[1] + ul_y), (l2[2] + ul_x, l2[3] + ul_y), (0,0,255), 3, cv.LINE_AA)
        
        # draw rectangle to show cropped img
        cv.rectangle(src, (ul_x,ul_y), (br_x, br_y), (0, 255, 0), 2)

        return (avgx + ul_x, hc//2 + ul_y)