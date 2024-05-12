#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

from safety_controller.visualization_tools import VisualizationTools

class SafetyController(Node):

    def __init__(self):
        super().__init__("safety_controller")
        # Declare parameters to make them available for use
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/vesc/high_level/ackermann_cmd")
        self.declare_parameter("stop_topic", "/vesc/low_level/input/safety")
        self.declare_parameter('odom_topic', "/pf/pose/odom")
        self.speed = 1.0

        # Fetch constants from the ROS parameter server
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.STOP_TOPIC = self.get_parameter('stop_topic').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value


        self.DRIVE_TOPIC = "/vesc/high_level/ackermann_cmd"

        self.line_pub = self.create_publisher(Marker, "/wall", 1)
        self.inner_circle_pub = self.create_publisher(Marker, "/inner_circle", 1)
        self.outer_circle_pub = self.create_publisher(Marker, "/outer_circle", 1)

        self.drive_msg = None

        self.car_x = 0
        self.car_y = 0

        # Initialize your publishers and subscribers here
        self.publisher_ = self.create_publisher(AckermannDriveStamped, self.STOP_TOPIC, 10)
        self.subscription = self.create_subscription(
           LaserScan,
           self.SCAN_TOPIC,
           self.lidar_callback,
           10)
        self.subscription2 = self.create_subscription(
           AckermannDriveStamped,
           self.DRIVE_TOPIC,
           self.drive_callback,
           10)
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            1
        )
        

        self.CAR_LENGTH = 0.3 # meters
        self.CAR_WIDTH = 0.32 # meters (includes buffer)
        self.DIST_TO_BUMPER = 0.12
        self.stop_count = 0

        self.pedestrian_zone_lower = np.array([[-16, 10.5],
                                                [-14, 11.5],
                                                [-14, 15],
                                                [-10, 24],
                                                [-36, 32],
                                                [-57, 31],
                                                [-57, 24]])
        self.pedestrian_zone_upper = np.array([[-14, 15],
                                                [-10.5, 15],
                                                [-2, 24],
                                                [-2, 27],
                                                [-30, 36],
                                                [-51, 36],
                                                [-52, 30]])

        self.get_logger().info("PEDESTRIAN AVOIDER STARTED!")

    # Write your callback functions here 

    def in_bounds(self, lower_bounds, upper_bounds, x, y):
        x_in = (lower_bounds[:, 0] <= x) & (x <= upper_bounds[:, 0])
        y_in = (lower_bounds[:, 1] <= y) & (y <= upper_bounds[:, 1])
        return np.any(x_in & y_in)
    
    def in_pedestrian_zone(self, x, y):
        return self.in_bounds(self.pedestrian_zone_lower, self.pedestrian_zone_upper, x, y)

    # intercept latest drive command
    def drive_callback(self, drive):
        # self.get_logger().info("Drive speed %s" % drive.drive.speed)
        self.drive_msg = drive

    def stop(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        if self.drive_msg:
            msg.drive.steering_angle = self.drive_msg.drive.steering_angle
        self.publisher_.publish(msg)


    def odom_callback(self, msg):
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y

    def lidar_callback(self, scan):
        # self.get_logger().info("Received scan")
        if self.speed < 0: return
        # if not self.in_pedestrian_zone(self.car_x, self.car_y): return
        # self.get_logger().info("PEDESTRIAN ZONE")


        steering_angle = 0.03675

        if self.drive_msg is not None:
            self.speed = self.drive_msg.drive.speed
            steering_angle += self.drive_msg.drive.steering_angle
        

        # found via experimenting
        # stop_dist = (0.41 * self.speed + 0.2) ** 2
        # stop_dist = 0.9
        stop_dist = 0.4 * self.speed - 0.1
        if self.speed < 1.0:
            stop_dist = 0.2 * self.speed + 0.1

        stop_dist += 0.2 # safety buffer

        # polar lidar coordinates to cartesian coordinates
        ranges = np.array(scan.ranges)
        angle_min = scan.angle_min
        angle_max = scan.angle_max 
        angle_increment = scan.angle_increment

        # angles = np.array([angle_min + angle_increment * i for i in range(len(ranges))])
        angles = np.linspace(angle_min, angle_max, len(ranges))

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        # create rectangle in front of car
        low_y = - self.CAR_WIDTH / 2 + x * np.arctan(steering_angle)
        high_y = self.CAR_WIDTH / 2 + x * np.arctan(steering_angle)

        low_x = self.DIST_TO_BUMPER + 0.03
        high_x = low_x + stop_dist

        # check if any points in rectangle
        within_front = np.where(np.logical_and(np.greater_equal(y, low_y), np.less_equal(y, high_y)))
        x_within = x[within_front]
        close_front = np.logical_and(x_within >= low_x, x_within <= high_x)

        # if np.any(close_front):
        # self.get_logger().info("close %s" % (np.count_nonzero(close_front)))
        # self.get_logger().info("front %s" % (len(x_within)))
        if np.count_nonzero(close_front) > 0.3 * len(x_within): 
            self.get_logger().info("STOP")
            self.stop_count += 1
            if self.stop_count > 2 * self.speed + 1:
                self.stop()
            return
        
        self.stop_count = 0
            
        return

        #######################################
        # DONUT SAFETY CONTROLLER NOT WORKING #
        #######################################

        # create donut section in front of car
        turn_radius = np.abs(self.CAR_LENGTH / np.sin(steering_angle))
        turn_sign = np.sign(steering_angle) # -1 for right, 1 for left turn
        inner_radius = turn_radius - self.CAR_WIDTH / 2.0
        outer_radius = turn_radius + self.CAR_WIDTH / 2.0

        # filter out points past 90 degree turn
        # x = x[np.where(abs(y) <= turn_radius)]
        # y = y[np.where(abs(y) <= turn_radius)]

        # y values
        inner_circle = turn_sign * inner_radius - turn_sign * np.sqrt(inner_radius**2 - x**2)
        outer_circle = turn_sign * outer_radius - turn_sign * np.sqrt(outer_radius**2 - x**2)

        # x values
        inner_circle = np.sqrt(inner_radius ** 2 - (y - turn_sign * turn_radius) ** 2) # + self.DIST_TO_BUMPER
        outer_circle = np.sqrt(outer_radius ** 2 - (y - turn_sign * turn_radius) ** 2) # + self.DIST_TO_BUMPER

        # self.get_logger().info('inner "%s"' % inner_circle)
        # self.get_logger().info('outer "%s"' % outer_circle)

        # end of the donut section
        dist_line = - turn_sign * np.tan(np.pi / 2 - stop_dist / turn_radius) * x + turn_sign * turn_radius
        # dist_line = (y - turn_sign * turn_radius)/(- turn_sign * np.tan(np.pi / 2 - stop_dist / turn_radius))

        # if distance line is past 90 degs
        if np.tan(np.pi / 2 - stop_dist / turn_radius) != turn_sign:
            dist_line = turn_sign * turn_radius * np.ones(x.shape)

        upper_side = outer_circle
        lower_side = inner_circle

        out_of_range = np.logical_or(np.isnan(lower_side), np.isnan(upper_side))
        in_range = np.logical_not(out_of_range)

        x = x[np.where(in_range)]
        y = y[np.where(in_range)]

        lower_side = lower_side[np.where(in_range)]
        upper_side = upper_side[np.where(in_range)]
        dist_line = dist_line[np.where(in_range)]

        # self.get_logger().info('"%s"' % lower_side)
            
        # check for any points in donut
        within_turn = np.where(np.logical_and(lower_side <= x, x <= upper_side))
        x_within = x[within_turn]
        y_within = y[within_turn]
        # self.get_logger().info('"%s"' % len(y_within))
    
        dist_line = dist_line = - turn_sign * np.tan(np.pi / 2 - stop_dist / turn_radius) * x_within + turn_sign * turn_radius
        # close_front = y_within <= dist_line
        
        # if turn_sign == -1: close_front = y_within >= dist_line

        close_front = x_within <= dist_line

        # VisualizationTools.plot_line(x, dist_line, self.line_pub, frame="laser")
        VisualizationTools.plot_line(x_within, dist_line, self.line_pub, frame="laser")
        VisualizationTools.plot_line(np.sqrt(inner_radius ** 2 - (y_within - turn_sign * turn_radius) ** 2), y_within, self.inner_circle_pub, frame="laser")
        VisualizationTools.plot_line(np.sqrt(outer_radius ** 2 - (y - turn_sign * turn_radius) ** 2), y, self.outer_circle_pub, frame="laser")

        # stop car if any points in donut
        if np.any(close_front):
            self.stop()

def main():

    rclpy.init()
    safety_controller = SafetyController()
    rclpy.spin(safety_controller)
    safety_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
