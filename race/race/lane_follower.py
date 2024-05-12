import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, PointStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32
from wall_follower.visualization_tools import VisualizationTools
import numpy as np

from vs_msgs.msg import ConeLocation


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('drive_topic', "default")

        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.driving = self.create_timer(0.05, self.timer_callback)

        self.lookahead = 1  # FILL IN #
        self.speed = 5.0  # FILL IN #
        self.wheelbase_length = 0.3  # FILL IN #

        # Adjust lookahead based on angle to lookahead point
        # higher angle error ~ lower lookahead distance
        # self.min_lookahead = 1.0 
        # self.max_lookahead = 2.0 

        # self.speed_to_lookahead = 2.0
        
        # # the angle to target s.t. the lookahead will be at its minimum
        # self.min_lookahead_angle = np.deg2rad(90) 

        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        
     
        self.target_pub = self.create_publisher(Marker, "/target_point", 1)
        self.radius_pub = self.create_publisher(Marker, "/radius", 1)

        self.max_steer = 0.34

        self.angle = None

        self.create_subscription(ConeLocation, "/relative_cone", 
            self.lane_callback, 1)
        
        self.get_logger().info("Lane Follower Initialized")

    def timer_callback(self):
        if self.angle is None: return
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.speed
        drive_msg.drive.steering_angle = self.angle
        self.drive_pub.publish(drive_msg)


    def lane_callback(self, target_msg):

        # convert target point to the car's frame
        car_to_target_x, car_to_target_y = target_msg.x_pos, target_msg.y_pos
 
        # angle to target point
        angle_error = np.arctan2(car_to_target_y, car_to_target_x)


        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.speed

        steer_angle = np.arctan2((self.wheelbase_length*np.sin(angle_error)), 
                                 0.5*self.lookahead + self.wheelbase_length*np.cos(angle_error))
        
        steer_angle = np.clip(steer_angle, -self.max_steer, self.max_steer)
        drive_msg.drive.steering_angle = steer_angle - 0.03675
        self.angle = steer_angle - 0.03675
        self.drive_pub.publish(drive_msg)


    

def main(args=None):

    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()