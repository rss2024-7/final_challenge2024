import rclpy
from rclpy.node import Node

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from .traffic_light import detect_red, detect_green


class LightDetector(Node):
    def __init__(self): 
        super().__init__("light_detector")
        self.publisher = self.create_publisher(Bool, "/light_detector", 10)
        self.debug_pub = self.create_publisher(Image, "/cone_debug_img", 10)
        self.subscriber = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.callback, 5)
        self.traffic_zone_sub = self.create_subscription(Bool, "/traffic_zone", self.traffic_zone_callback, 1)
        self.bridge = CvBridge()

        self.create_timer(0.1, self.timer_callback)

        self.in_traffic_zone = False
        self.img_msg = None

        self.get_logger().info("Light Detector Initialized")

    def traffic_zone_callback(self, msg):
        self.get_logger().info(f"Traffic Cb: {msg.data}")
        self.in_traffic_zone = msg.data

    def timer_callback(self):
        if self.img_msg is None: return
        if not self.in_traffic_zone:
            self.debug_pub.publish(self.img_msg) 
            return

        self.get_logger().info("waiting for green...")

        # Process image with CV Bridge
        image = self.bridge.imgmsg_to_cv2(self.img_msg, "bgr8")

        boolean, cntr, rad = detect_green(np.asarray(image[:, :]))

        pub = Bool()
        pub.data = boolean
        self.publisher.publish(pub)

        if boolean:
            self.get_logger().info(f"radius = {rad}, area = {np.pi * rad**2}")

        image = cv.circle(image, (int(cntr[0]), int(cntr[1])), radius = int(rad), color = (255, 0, 0), thickness = -1)

        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)

    def callback(self, img_msg):
        self.img_msg = img_msg




def main(args=None):
    rclpy.init(args=args)
    detector = LightDetector()
    rclpy.spin(detector)
    rclpy.shutdown()

if __name__=="__main__":
    main()