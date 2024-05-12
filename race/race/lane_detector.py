import rclpy
from rclpy.node import Node
import numpy as np

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from vs_msgs.msg import ConeLocationPixel

import math

# import your color segmentation algorithm; call this function in ros_image_callback!
from .line_detection import get_target_pix


class LaneDetector(Node):
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    def __init__(self):
        super().__init__("lane_detector")
        # toggle line follower vs cone parker
        self.LineFollower = False

        # Subscribe to ZED camera RGB frames
        self.cone_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 10)
        self.debug_pub = self.create_publisher(Image, "/cone_debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

        self.get_logger().info("Lane Detector Initialized")

        self.pursuit_point = None
        self.bisector_line = None
        self.left_line_bias = 0.55 # 0.60

        # init left line
        rho = 288
        theta = 0.80285
        a = math.cos(theta)
        b = math.sin(theta) 
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        self.left_line = (rho, theta, a, b, x0, y0, pt1, pt2)

        # init right line
        rho = 55
        theta = 1.850049
        a = math.cos(theta)
        b = math.sin(theta) 
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        self.right_line = (rho, theta, a, b, x0, y0, pt1, pt2)




    def get_intersect(self, a1, a2, b1, b2):
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

    def image_callback(self, image_msg):
        # Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        # From your bounding box, take the center pixel on the bottom
        # (We know this pixel corresponds to a point on the ground plane)
        # publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        # convert it to the car frame.

        #################################
        # YOUR CODE HERE
        # detect the cone and publish its
        # pixel location in the image.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #################################  




        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")


        shape = image.shape
        pub = ConeLocationPixel()

        if not self.pursuit_point:
            self.pursuit_point = (int(0.575*shape[1]), int(0.45*shape[0]))

        left_line, right_line, frame = get_target_pix(np.asarray(image[:, :])) 
        # self.get_logger().info("left rho:" + str(left_line[0]))
        # self.get_logger().info("left theta:" + str(left_line[1]))
        # self.get_logger().info("left a:" + str(left_line[2]))
        # self.get_logger().info("left b:" + str(left_line[3]))
        # self.get_logger().info("left x0:" + str(left_line[4]))
        # self.get_logger().info("left y0:" + str(left_line[5]))
        # self.get_logger().info("left pt1:" + str(left_line[6]))
        # self.get_logger().info("left pt2:" + str(left_line[7]))


        # self.get_logger().info("right rho:" + str(right_line[0]))
        # self.get_logger().info("right theta:" + str(right_line[1]))
        # self.get_logger().info("right a:" + str(right_line[2]))
        # self.get_logger().info("right b:" + str(right_line[3]))
        # self.get_logger().info("right x0:" + str(right_line[4]))
        # self.get_logger().info("right y0:" + str(right_line[5]))
        # self.get_logger().info("right pt1:" + str(right_line[6]))
        # self.get_logger().info("right pt2:" + str(right_line[7]))

        if left_line:
            self.left_line = left_line
        # else:
        #     left_line = self.left_line
        
        if right_line:
            self.right_line = right_line
        # else:
        #     right_line = self.right_line
        
        intersection = self.get_intersect(self.left_line[6], self.left_line[7], self.right_line[6], self.right_line[7])
        if intersection:
            bisector_theta = self.left_line_bias*self.left_line[1] + (1-self.left_line_bias)*self.right_line[1] + np.pi/2
            x,y = intersection
            rho = self.left_line[0]
            a = math.cos(bisector_theta)
            b = math.sin(bisector_theta)
            pt1 = (x,y)
            pt2 = (int(x - 1000*(-b)), int(y - 1000*(a)))

            self.bisector_line = (rho, bisector_theta, a, b, x, y, pt1, pt2)
            self.pursuit_point = (int(x - 50*(-b)), int(y - 50*(a)))
        # else:
        #     pursuit_point = self.pursuit_point

        cv.line(frame, self.left_line[6], self.left_line[7], (255,0,0), 3, cv.LINE_AA)
        cv.line(frame, self.right_line[6], self.right_line[7], (0,0,255), 3, cv.LINE_AA)
        cv.line(frame, self.bisector_line[6], self.bisector_line[7], (255,0,255), 3, cv.LINE_AA)
        cv.circle(frame, self.pursuit_point, 5, (255,255,0),5) 

        pub.u = float(self.pursuit_point[0])
        pub.v = float(self.pursuit_point[1])
        debug_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8") 


        # if left_line:
        #     self.get_logger().info("left theta:" + str(left_line[1]))
        # if right_line:
        #     self.get_logger().info("right theta:" + str(right_line[1])) 
        
        self.debug_pub.publish(debug_msg)
        self.cone_pub.publish(pub)

def main(args=None):
    rclpy.init(args=args)
    lane_detector = LaneDetector()
    rclpy.spin(lane_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()