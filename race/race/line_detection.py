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
      
def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """

    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return None
    return (int(x/z), int(y/z))


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)

def get_target_pix(src):
    """	
    Find center pixel of detected lane
	Input:
		src: np.3darray; the input image from camera. BGR.
	Return:
		target_pixel: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
    """
    
    frame = src.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray,(5,5),0)
    shape = gray.shape
    
    mask = np.zeros(shape, dtype="uint8")
    
    ul = [int(0.2*shape[1]), int(0.45*shape[0])]
    ur = [int(0.95*shape[1]), int(0.45*shape[0])]
    bl = [int(0.05*shape[1]),int(0.7*shape[0])]
    br = [int(0.98*shape[1]),int(0.7*shape[0])]

    pts = np.array([ul, ur, br, bl], np.int32)
    pts = pts.reshape((-1,1,2))
    mask = cv.fillPoly(mask,[pts],255)
    crop = cv.bitwise_and(gray, gray, mask=mask)

    thresh  = cv.threshold(crop,170,255,cv.THRESH_BINARY)[1] # 150 -->  170

    edges = cv.Canny(thresh, 50, 200)

    lines = cv.HoughLines(edges, 1, np.pi / 180, 50, None, 0, 0) #70 --> 50
    # lines = cv.HoughLines(edges, 1, np.pi/180, 200)

    intersection = None

    left_line = None
    right_line = None
    bisector_line = None
    pursuit_point = None
    
    left_line = None
    right_line = None
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta) 
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

            if (not left_line and theta > 0.261799 and theta < 1.309): # 0.2
                left_line = (rho, theta, a, b, x0, y0, pt1, pt2)
            elif (not right_line and theta > 1.8326 and theta < 2.87979): #1.8326 -->1.7326
                right_line = (rho, theta, a, b, x0, y0, pt1, pt2)

            # left_line_bias = 0.70

    frame = cv.polylines(frame, [pts], True, (0,255,0), 2)
    return left_line, right_line, frame