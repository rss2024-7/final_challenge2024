import rclpy
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
from sensor_msgs.msg import Image
from .detector import StopSignDetector, draw_rect
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
import time

class SignDetector(Node):
    def __init__(self):
        super().__init__("stop_detector")
        self.detector = StopSignDetector(0.3) #0.1
        self.publisher = None #TODO
        self.subscriber = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.callback, 10)
        self.debug_pub = self.create_publisher(Image, "/stop_img", 10)
        self.bridge = CvBridge()

        self.stop_pub = self.create_publisher(Bool,
                                "/stopsign",
                                1)
        
        self.create_timer(0.05, self.timer_callback)
        

        self.blind_start = 0

        self.img_msg = None

        self.get_logger().info("Stop Detector Initialized")

    def timer_callback(self):

        if self.img_msg is None: return
        if time.time() - self.blind_start < 10: return

        image = self.bridge.imgmsg_to_cv2(self.img_msg, "bgr8")

        # image = cv2.resize(image, (0,0), fx = 0.5, fy = 0.5)

        predict_start = time.time()
        has_stop_sign, bounding_box = self.detector.predict(image)
        # self.get_logger().info(f"{time.time() - predict_start}")

        if not has_stop_sign: 
            # debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8") 
            # self.debug_pub.publish(debug_msg)
            # self.get_logger().info("No stop sign")
            return
        
        self.get_logger().info("stop sign detected")
        
        if bounding_box == (0, 0, 0, 0): return

        x_min, y_min, x_max, y_max = bounding_box

        area = (x_max - x_min) * (y_max - y_min)

        # self.get_logger().info(f"{area=}")

        

        stop = area >= 4000
        stop = area >= 100

        # if not stop: return

        msg = Bool()
        msg.data = True
        self.stop_pub.publish(msg)

        self.blind_start = time.time()

    def callback(self, img_msg):
        # return
        # return
        # if time.time() - self.blind_start < 3: return
        self.img_msg = img_msg
        return
        # Process image with CV Bridge
        # self.get_logger().info("image received")
        image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        image = cv.resize(image, (0,0), fx = 0.1, fy = 0.1)
        
        
        # shape = image.shape
        # mask = np.zeros(shape[:2], dtype="uint8")
        mask_prop = 1/2
        # mask = cv2.rectangle(mask, (0, 0), (int(shape[1]), int(shape[0])), 255, -1)
        
        # image = cv2.bitwise_and(image, image, mask=mask)

        #TODO: 

        # self.get_logger().info("predicting")
        has_stop_sign, bounding_box = self.detector.predict(image)
        # self.get_logger().info(f"{has_stop_sign=}")

        if not has_stop_sign: 
            # debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8") 
            # self.debug_pub.publish(debug_msg)
            # self.get_logger().info("No stop sign")
            return
        
        self.get_logger().info("stop sign detected")
        
        if bounding_box == (0, 0, 0, 0): return

        x_min, y_min, x_max, y_max = bounding_box

        area = (x_max - x_min) * (y_max - y_min)

        # self.get_logger().info(f"{area=}")

        

        stop = area >= 4000

        msg = Bool()
        msg.data = True
        self.stop_pub.publish(msg)

        self.blind_start = time.time()

        # stop_image = cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

        # debug_msg = self.bridge.cv2_to_imgmsg(stop_image, "bgr8") 
        # self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    detector = SignDetector()
    rclpy.spin(detector)
    rclpy.shutdown()

if __name__=="__main__":
    main()
