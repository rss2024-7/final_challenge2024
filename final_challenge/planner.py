"""
Add the following to dockerfile:

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install drake
RUN python3 -m pip install opencv-python


To test:
ros2 launch racecar_simulator simulate.launch.xml

ros2 launch final_challenge planner.launch.xml
"""


import rclpy
from rclpy.node import Node

assert rclpy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point32
from geometry_msgs.msg import Point as ROSPoint
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration
from .utils import LineTrajectory

import time
import numpy as np
import pydot

from nominal_path import nominal_path


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )
        
        # FOR SIMULATION TESTING
        self.odom_sub = self.create_subscription(
            Odometry, 
            self.odom_topic,
            self.odom_cb,
            1
        )

        self.closest_pt_pub = self.create_publisher(Marker, "viz/closest_pt", 1)
        self.shell_pub = self.create_publisher(Marker, "viz/shell_pub", 1)

        self.current_pose = None
        self.goal_pose = None
        self.map = None
        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.get_logger().info("=============================READY=============================")


    def map_cb(self, msg):
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id
        map_width = msg.info.width
        map_height = msg.info.height
        map_resolution = msg.info.resolution  # resolution in meters/cell
        map_data = msg.data


    def odom_cb(self, msg):
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id
        position_x = msg.pose.pose.position.x
        position_y = msg.pose.pose.position.y
        position_z = msg.pose.pose.position.z
        orientation_x = msg.pose.pose.orientation.x
        orientation_y = msg.pose.pose.orientation.y
        orientation_z = msg.pose.pose.orientation.z
        orientation_w = msg.pose.pose.orientation.w
        theta = 2 * np.arctan2(orientation_z, orientation_w)
        self.current_pose = np.array([position_x, position_y, theta])

        # print(f"odom current_pose: {self.current_pose}")


    def pose_cb(self, msg):
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id
        position_x = msg.pose.pose.position.x
        position_y = msg.pose.pose.position.y
        position_z = msg.pose.pose.position.z
        orientation_x = msg.pose.pose.orientation.x
        orientation_y = msg.pose.pose.orientation.y
        orientation_z = msg.pose.pose.orientation.z
        orientation_w = msg.pose.pose.orientation.w
        theta = 2 * np.arctan2(orientation_z, orientation_w)
        self.current_pose = np.array([position_x, position_y, theta])

        print(f"current_pose: {self.current_pose}")


    def goal_cb(self, msg):
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id
        position_x = msg.pose.position.x
        position_y = msg.pose.position.y
        position_z = msg.pose.position.z
        orientation_x = msg.pose.orientation.x
        orientation_y = msg.pose.orientation.y
        orientation_z = msg.pose.orientation.z
        orientation_w = msg.pose.orientation.w
        theta = 2 * np.arctan2(orientation_z, orientation_w)
        self.goal_pose = np.array([position_x, position_y, theta])

        print(f"shell_pose set: {self.goal_pose}")

        self.plan_path(self.goal_pose, self.map)


    def publish_point(self, point, publisher, r, g, b):
        self.get_logger().info("Before Publishing point")
        if self.start_pub.get_subscription_count() > 0:
            self.get_logger().info("Publishing point")
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = 0
            marker.type = 2  # sphere
            marker.action = 0
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0
            publisher.publish(marker)
        elif publisher.get_subscription_count() == 0:
            self.get_logger().info("Not publishing point, no subscribers")
    

    def plan_path(self, shell_point, map):
        self.get_logger().info(f"PLANNING PATH DEVIATION TO {shell_point}")

        goal_pos = shell_point[:2]

        # Visualize shell markers
        self.publish_point(goal_pos, self.end_pub, 0.0, 1.0, 0.0)
        
        # Find closes point on path to the designated point
        # First, find closest endpoint to the designated point
        closest_dist_sq = np.inf
        closest_idx = 0
        for i in range(len(nominal_path["points"])):
            new_dist_sq = (nominal_path["points"][i]["x"] - goal_pos[0])**2 + (nominal_path["points"][i]["y"] - goal_pos[1])**2
            if new_dist_sq < closest_dist_sq:
                closest_dist_sq = new_dist_sq
                closest_idx = i
            
        # Now, search the two segments adjacent to the closest endpoint to find
        # the true closest point to the designated point
        closest_dist_sq = np.inf
        closest_pt = None
        for i in range(closest_idx-1, closest_idx+1):
            if i < 0 or i >= len(nominal_path["points"]):
                continue

            start = np.array([nominal_path["points"][i]["x"], nominal_path["points"][i]["y"]])
            end = np.array([nominal_path["points"][i+1]["x"], nominal_path["points"][i+1]["y"]])

            start_to_point = shell_point - start
            start_to_end = end - start

            segment_length_squared = np.dot(start_to_end, start_to_end)
            
            projection = np.dot(start_to_point, start_to_end) / segment_length_squared

            # Clamp the projection parameter to the range [0, 1]
            projection = max(0, min(1, projection))
            closest_pt_estimate = start + projection * start_to_end
            closest_pt_estimate_dist = np.linalg.norm(shell_point - closest_pt_estimate)


            if (closest_pt_estimate_dist < closest_dist_sq) {
                closest_dist_sq = closest_pt_estimate_dist
                closest_pt = closest_pt_estimate
            }
        
        self.publish_point(closest_pt, self.closest_pt_pub, 1.0, 0.5, 0.0)


            

        ANGLE_INCREMENT = 0.1  # radians
        for angle in np.arange(0, 2*np.pi, ANGLE_INCREMENT):
            pass
            

        

        # traj_pose_array = PoseArray()
        # length_sum = 0.0
        # previous_point = None
        # for t in np.linspace(traj.start_time(), traj.end_time(), 100):
        #     self.get_logger().info(f"{traj.value(t)}")

        #     pose = Pose()
        #     pose.position.x = float(traj.value(t)[0,0])
        #     pose.position.y = float(traj.value(t)[1,0])
        #     pose.position.z = 0.0  # Assuming z is 0 for 2D coordinates
        #     pose.orientation.w = 1.0  # Neutral orientation
        #     traj_pose_array.poses.append(pose)

        #     current_point = np.array([pose.position.x, pose.position.y])

        #     # Calculate distance from the previous point if it exists
        #     if previous_point is not None:
        #         distance = np.linalg.norm(current_point - previous_point)
        #         length_sum += distance

        #     # Update previous_point to the current point for the next iteration
        #     previous_point = current_point

        # # set frame so visualization works
        # traj_pose_array.header.frame_id = "/map"  # replace with your frame id

        # self.traj_pub.publish(traj_pose_array)

        # self.get_logger().info(f"Total length of the trajectory: {length_sum}")


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
