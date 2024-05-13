import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, Odometry
from .utils import LineTrajectory
from .rrt import RRT

from std_msgs.msg import ColorRGBA, Float32, Bool

from ament_index_python.packages import get_package_share_directory

import time

import numpy as np
import cv2
import sys
import os


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

        self.get_logger().info(f"{self.odom_topic=}")

        self.shell_radius = 3  # meters
        self.dist_from_shell = 0.2  # meters


        self.shell_paths = []
        self.shell_collected = True

        self.turn_zone_lower = np.array([[-16, 23], # narrow hallway 1
                                         [-22, 29], # narrow hallway 2
                                         [-49, 30], # between narrow hallway and vending machine hallway
                                         [-58, 26], # between vending machine hallway crosswalks
                                         [-58, 5], # vending machine hallway
                                        ])
        self.turn_zone_upper = np.array([[-11, 28], # narrow hallway 1
                                         [-19, 32], # narrow hallway 2
                                         [-34, 37], # between narrow hallway and vending machine hallway
                                         [-51, 34], # between vending machine hallway crosswalks
                                         [-51, 24], # vending machine hallway
                                        ])
        
        # self.endzone_lower = np.array([[-25, -5],
        #                                [-60, -5],])
        # self.endzone_upper = np.array([[-16, 2],
        #                                [-51, 7],]) # -51, 2 before
        
        self.endzone_lower = np.array([[-60, -5],])
        self.endzone_upper = np.array([[-51, 7],]) # -51, 2 before


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

        self.tree_pub = self.create_publisher(
            Marker,
            "/tree",
            20
        )

        self.debug_pub = self.create_publisher(
            MarkerArray,
            "/debug",
            20
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.shell_path,
            1
        )

        self.shell_sub = self.create_subscription(PointStamped,
                                            "/clicked_point",
                                            self.shell_callback,
                                            3)

        # parent_directory = os.path.abspath('..')
        # sys.path.append(parent_directory)
        # script_dir = sys.path[0]
        # img_path = os.path.join(script_dir, '../maps/stata_basement_dilated_invert.png')

        # padded_map = cv2.imread('stata_basement.png', cv2.IMREAD_GRAYSCALE)
        # self.padded_map_grid = np.asanyarray(padded_map, order='C')

        
        save_prefix = os.path.join(os.environ["HOME"], "lab6_trajectories")

        if not os.path.exists(save_prefix):
            self.get_logger().info("Creating the trajectory logging directory: {}".format(save_prefix))
            os.makedirs(save_prefix)

        self.save_path = os.path.join(save_prefix,
                                      time.strftime("%Y-%m-%d-%H-%M-%S") + ".traj")

        self.current_pose = None
        self.goal_pose = None
        self.map = None
        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.map_size = None

        x = 25.900000
        y = 48.50000
        theta = 3.14
        resolution = 0.0504
        self.transform = np.array([[np.cos(theta), -np.sin(theta), x],
                    [np.sin(theta), np.cos(theta), y],
                    [0,0,1]])


        self.nodes_coords = []

        self.declare_parameter("lane", "default")
        path = self.get_parameter("lane").get_parameter_value().string_value
        self.lane_traj = LineTrajectory(self, "/lane")
        self.lane_traj.load(path)

        self.shell_points = []

        map_path = os.path.join(get_package_share_directory('city'),'maps/custom_map.png')

        self.map_data = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

        self.map_data = np.clip(self.map_data, 0,100)

        self.map_data = self.map_data.flatten("C")

        self.map_resolution = 0.0504
        self.map_width = 1730
        self.map_height = 1300

        # points = self.lane_traj.points
        # for i in range(len(points) - 1):
        #     self.edit_map(np.array(points[i]), np.array(points[i + 1]))

        self.map_info = (self.map_data, self.map_width, self.map_height, self.map_resolution) #if needed for passing into RRT object

        self.RRT_planner = RRT(self.map_info)

        self.shell_pub = self.create_publisher(MarkerArray, "/shell_point", 3)
        # self.shell_near_pub = self.create_publisher(Marker, "/shell_near_point", 3)

        self.turn_pub = self.create_publisher(
            Float32,
            "/turnaround",
            1
        )
        self.turn_success_sub = self.create_subscription(
            Bool,
            "/turn_outcome",
            self.turn_success_callback,
            1
        )

        self.shell_collected_sub = self.create_subscription(
            Bool,
            "/shell_collected",
            self.shell_collected_callback,
            3
        )

        self.initialized = False

        self.turning = False

        self.car_side = 75

        self.car_in_endzone = True

        self.first_turn_cooldown = time.time()

        # self.out_of_start = False

        self.get_logger().info("------READY-----")

    def shell_collected_callback(self, msg):
        self.get_logger().info("Collected shell")
        self.shell_collected = True


    def in_bounds(self, lower_bounds, upper_bounds, x, y):
        x_in = (lower_bounds[:, 0] <= x) & (x <= upper_bounds[:, 0])
        y_in = (lower_bounds[:, 1] <= y) & (y <= upper_bounds[:, 1])

        return np.any(x_in & y_in)

    def in_turn_zone(self, x, y):
        return self.in_bounds(self.turn_zone_lower, self.turn_zone_upper, x, y)
    
    def in_endzone(self, x, y):
        return self.in_bounds(self.endzone_lower, self.endzone_upper, x, y)

    def turn_success_callback(self, msg):
        if msg.data:
            self.get_logger().info("switch")
            if self.car_side == 75:
                self.car_side = 25
            else:
                self.car_side = 75
            # self.get_logger().info(f"{self.car_side}")
        self.turning = False
    def turn_around(self):
        if self.turning: return
        if not self.shell_collected: return

        self.get_logger().info("turning around...")
        
        turn_msg = Float32()
        turn_msg.data = 420.69
        self.turn_pub.publish(turn_msg)
        self.turning = True
        # time.sleep(5)

    def shell_path(self, msg):

        car_pos_x = msg.pose.pose.position.x
        car_pos_y = msg.pose.pose.position.y

        car_pos = np.array([car_pos_x, car_pos_y]) 

        # self.get_logger().info(f'{self.shell_points=}')

        if self.in_endzone(car_pos_x, car_pos_y):
            if self.car_in_endzone: return
            self.get_logger().info("turning around in end zone")
            # if not self.out_of_start: return
            # if time.time() - self.start_cooldown < 30: return
            self.car_in_endzone = True
            self.get_logger().info("endzone turn")
            self.turn_around()
            return
        
        self.car_in_endzone = False
        # if not self.out_of_start:
        #     self.start_cooldown = time.time()
        #     self.out_of_start = True

        if not self.initialized: return

        if len(self.shell_points) == 0: 
            if not self.in_turn_zone(car_pos_x, car_pos_y): return

            # turn back towards start point
            car_side = self.car_side
            if car_side == 75:
                self.get_logger().info("turn towards start")
                self.turn_around()

            return
        
        # previous shell has not been collected
        if not self.shell_collected: return 
        
        # self.get_logger().info(f'{self.shell_points[0]}')
        shell_point, shell_side = self.shell_points[0]

        shell_side = int(shell_side)
        car_side = self.car_side

        points = np.array(self.lane_traj.points)
        traj_x = points[:, 0]
        traj_y = points[:, 1]
            
        car_index = self.find_closest_segment(traj_x, traj_y, car_pos_x, car_pos_y)
        shell_index = self.find_closest_segment(traj_x, traj_y, shell_point[0], shell_point[1])
        
        if shell_side != car_side:
            # if not self.in_turn_zone(car_pos_x, car_pos_y): return

            # self.get_logger().info(f"{car_index=}, {shell_index=}, {car_side=}, {shell_side=}")
            # self.get_logger().info(f"{(car_index > shell_index)=}, {(car_side > shell_side)=}")
            # if car_index == shell_index: return
            if (car_index > shell_index) == (car_side > shell_side):
                # self.get_logger().info("Turn")
                self.get_logger().info("turn for shell on other side")
                self.turn_around()

            return
        
        # if car_side == 25 and shell_index > car_index:
        #     self.get_logger().info(f"u turn for point behind. same inner lane {shell_index} {car_index}")
        #     self.turn_around()
        #     return
        # if car_side == 75 and shell_index < car_index:
        #     self.get_logger().info(f"u turn for point behind. same outer lane. {shell_index} {car_index}")
        #     self.turn_around()
        #     return       

        
        if np.linalg.norm(car_pos - self.shell_points[0][0]) > self.shell_radius:
            return

        # self.get_logger().info("planning shell path...")
        # path = self.shell_paths.pop(0)

        # path = [tuple(row) for row in path]


        # self.trajectory.points = path
        # self.trajectory.save(self.save_path)
        # self.traj_pub.publish(self.trajectory.toPoseArray())
        # self.trajectory.publish_viz()

        # self.plan_path(car_pos, point, self.map)

        self.trajectory.clear()

        path = self.RRT_planner.plan_straight_line(car_pos, shell_point)

        # self.get_logger().info("attempted to plan...")

        if path is None:
            return
        
        self.get_logger().info("plan successful")
        
        self.shell_points.pop(0)

        path = [tuple(row) for row in path]

        self.trajectory.points = path
        self.trajectory.save(self.save_path)
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()
        self.shell_collected = False

    
    def find_closest_point(self, x, y):
        points = np.array(self.lane_traj.points)
        traj_x = points[:, 0]
        traj_y = points[:, 1]
      
        points = np.vstack((traj_x, traj_y)).T
        v = points[:-1, :] # segment start points
        w = points[1:, :] # segment end points

        p = np.array([x, y])
        
        l2 = np.sum((w - v)**2, axis=1)

        t = np.maximum(0, np.minimum(1, np.sum((p - v) * (w - v), axis=1) / l2))

        projections = v + t[:, np.newaxis] * (w - v)
        min_distances = np.linalg.norm(p - projections, axis=1)

        closest_segment_index = np.where(min_distances == np.min(min_distances))[0][0]

        
        start = v[closest_segment_index]
        end = w[closest_segment_index]
        p = np.array([x, y])

        line_length = np.linalg.norm(end - start)
        projection_proportion = np.dot(end - start, p - start) / line_length ** 2
        closest_point = start + projection_proportion * (end - start)

        dist = np.linalg.norm(closest_point - p)
        
        
        near_point = p + self.dist_from_shell / dist * (closest_point - p)

        return near_point
        
    def shell_callback(self, point_msg):
        near_point = self.find_closest_point(point_msg.point.x, point_msg.point.y)
        

        # self.get_logger().info(f'{point_msg.point.x=} {point_msg.point.y=}')
        # self.get_logger().info(f'{near_point=}')
        if self.real_to_map(near_point) != 75 and self.real_to_map(near_point) != 25:
            self.get_logger().info("Cannot Add Shell Here")
            return

        side = self.real_to_map(near_point)
        self.shell_points.append((near_point, side))

        self.get_logger().info(f"{side=}")

        if len(self.shell_points) >= 3: self.initialized = True

        self.pub_points([point for point, side in self.shell_points])

    def plan_path(self, start_point, end_point, map):

        self.trajectory.clear()

        self.get_logger().info("Planning path")

        path = self.RRT_planner.plan_path(start_point, end_point) #, np.array([x + 0.5*np.cos(theta), y + 0.5*np.sin(theta)]))
        self.get_logger().info("complete")

        if path is None:
            self.get_logger().info("PATH NOT POSSIBLE")
            return

        path = [tuple(row) for row in path]

        self.trajectory.points = path
        self.trajectory.save(self.save_path)
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

        # points = np.array(self.lane_traj.points)
        # traj_x = points[:, 0]
        # traj_y = points[:, 1]

        # closest_seg_index = self.find_closest_segment(traj_x, traj_y, near_point[0], near_point[1], side)
        # seg_start_x = traj_x[closest_seg_index]
        # seg_start_y = traj_y[closest_seg_index]
        # seg_end_x = traj_x[closest_seg_index + 1]
        # seg_end_y = traj_y[closest_seg_index + 1]
        # start = self.get_lookahead_point(seg_start_x, seg_start_y, seg_end_x, seg_end_y, near_point[0], near_point[1])

        # self.shell_paths.append(self.RRT_planner.plan_path(start, near_point))


    def find_closest_segment(self, x, y, car_x, car_y, side=None):
        """Finds closest line segment in trajectory and returns its index.
            Code based on https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment/1501725#1501725
            and modified to use numpy arrays for better speed

        Args:
            x (1D np array): x values of trajectory
            y (1D np array): y values of trajectorffy
            car_x (float): x position of car
            car_y (float): y position of car
        Returns:
            int: index of start of closest line segment in the trajectory arrays
        """
        if side is not None and side == 25:
            x = np.flip(x)
            y = np.flip(y)
        points = np.vstack((x, y)).T
        v = points[:-1, :] # segment start points
        w = points[1:, :] # segment end points
        p = np.array([[car_x, car_y]])
        
        l2 = np.sum((w - v)**2, axis=1)

        l2 += 10e-10

        t = np.maximum(0, np.minimum(1, np.sum((p - v) * (w - v), axis=1) / l2))

        projections = v + t[:, np.newaxis] * (w - v)
        min_distances = np.linalg.norm(p - projections, axis=1)

        # # if too close to end point of segment, take it out of consideration for closest line segment
        # start_point_distances = np.linalg.norm(v-p, axis=1)
        # min_distances[np.where(start_point_distances[:-1] < self.shell_radius)] += 500

        # if np.min(min_distances) > self.shell_radius:
        #     self.shell_radius = np.min(min_distances)

        closest_segment_index = np.where(min_distances == np.min(min_distances))

        if len(closest_segment_index) == 0: return 0
        if len(closest_segment_index[0]) == 0: return 0

        return closest_segment_index[0][0] if side != 25 else len(x) - closest_segment_index[0][0] - 1
    
    def get_lookahead_point(self, x1, y1, x2, y2, origin_x, origin_y):
        """Based on https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment/1501725#1501725.
            Finds lookahead point (intersection between circle and line segment).

        Args:
            x1 (float): line segment start x
            y1 (float): line segment start y
            x2 (float): line segment end x
            y2 (float): line segment end y
            origin_x (float): center of circle x
            origin_y (float): center of circle y

        Returns:
            1D np array of size 2: point of intersection (not necessarily on the line segment). 
            None if no intersection even if line segment is extended
        """
        Q = np.array([origin_x, origin_y])                  # Centre of circle
        r = self.shell_radius  # Radius of circle

        P1 = np.array([x1, y1])  # Start of line segment
        P2 = np.array([x2, y2])
        V = P2 - P1  # Vector along line segment
        a = np.dot(V, V)
        b = 2 * np.dot(V, P1 - Q)
        c = np.dot(P1, P1) + np.dot(Q, Q) - 2 * np.dot(P1, Q) - r**2

        disc = (b**2 - 4 * a * c)
        if disc < 0:
            return None
        
        sqrt_disc = np.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)

        t = min(t1, t2)

        if (t < 0): t = 0

        return P1 + t * V

    def real_to_map(self, point):
        return self.map_data[self.index_from_pixel(self.real_to_pixel(point))]

    def pose_cb(self, msg):
        # self.shell_points = []
        # gets the initial pose 
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


        self.get_logger().info("starting pose: " + np.array2string(self.current_pose))

    def goal_cb(self, msg):
        self.shell_points = []
        self.initialized = False
        return
        # gets the goal pose
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
        self.goal_pose = np.array([position_x, position_y, theta]) # ***

        
        self.get_logger().info("goal pose: " + np.array2string(self.goal_pose))

        if self.current_pose is None:
            self.get_logger().info("NO STARTING POINT")
            return
        if self.map is None:
            self.get_logger().info("NO MAP")
            return
        
        self.plan_path(self.current_pose, self.goal_pose, self.map)

    def pub_point(self, pub, rgb, point):
        color = ColorRGBA()
        color.r, color.g, color.b = rgb 
        color.a = 1.0

        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color = color
        marker.pose.orientation.w = 1.0
        marker.pose.position = Point(x=float(point[0]), y=float(point[1]), z=0.0)

        pub.publish(marker)

    def pub_points(self, points):
        color = ColorRGBA()
        color.r = 0.0
        color.g = 0.0
        color.b = 1.0
        color.a = 1.0

        marker_array = MarkerArray()
        
        marker_id = 0

        for point in points:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.25
            marker.scale.y = 0.25
            marker.scale.z = 0.25
            marker.color = color
            marker.pose.orientation.w = 1.0
            marker.pose.position = Point(x=float(point[0]), y=float(point[1]), z=0.0)
            marker.id = marker_id
            marker_id += 1

            marker_array.markers.append(marker)
        
        self.shell_pub.publish(marker_array)
        self.get_logger().info('points published')

    def edit_map(self, p1, p2): #where p1 and p2 are adjacent points in .traj
        vec = p2-p1
        dist = np.linalg.norm(vec)
        num_intervals = int(dist/0.01)

        for interval in range(1,num_intervals):
            coord = p1 + interval/num_intervals * vec
            pixel = self.real_to_pixel(coord)
            self.map_data[self.index_from_pixel(pixel)] = 100

    def map_cb(self, msg):
        return
        self.map = msg
        timestamp = msg.header.stamp
        frame_id = msg.header.frame_id
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution  # resolution in meters/cell
        self.map_data = msg.data

        points = self.lane_traj.points
        for i in range(len(points) - 1):
            self.edit_map(np.array(points[i]), np.array(points[i + 1]))

        # self.map_data = self.padded_map_grid
        self.map_orientation = msg.info.origin.orientation
        self.map_position = msg.info.origin.position
        self.map_info = (self.map_data, self.map_width, self.map_height, self.map_resolution)
        
        self.RRT_planner = RRT(self.map_info)

        self.get_logger().info("map height: " + str(self.map_height))
        self.get_logger().info("map width: " + str(self.map_width))
        self.get_logger().info("map data length: " + str(len(self.map_data)))
        self.get_logger().info("map [0][0]: " + str(self.pixel_to_real([0,0])))
        self.get_logger().info("map [0][map width]: " + str(self.pixel_to_real([0,self.map_width])))
        self.get_logger().info("map [map height][0]: " + str(self.pixel_to_real([self.map_height,0])))
        self.get_logger().info("map [map height][map width]: " + str(self.pixel_to_real([self.map_height,self.map_width])))
        self.get_logger().info("origin (25.9, 48.5) to pixel: " + str(self.real_to_pixel([25.9, 48.5])))



    def pixel_to_real(self, pixel, column_row=False):
        # if column_row = True, pixel format is [v (column index), u (row index)] 
        if column_row:
            pixel = np.array([pixel[0], pixel[1]]) 
        else:
            pixel = np.array([pixel[1], pixel[0]]) 

        pixel = pixel * self.map_resolution
        pixel = np.array([*pixel,1])
        pixel = np.linalg.inv(self.transform) @ pixel
        point = pixel

        # returns real x,y 
        return point

    def real_to_pixel(self, point, column_row=False):
        #takes in [x,y] real coordinate
        point = np.array([point[0], point[1], 1])
        point = self.transform @ point
        point = point / self.map_resolution
        pixel = np.floor(point)

        if column_row: # returns [v (column index), u (row index)] 
            return np.array([int(pixel[0]), int(pixel[1])])
        else:
            # returns [u (row index), v (column index)]
            return np.array([int(pixel[1]), int(pixel[0])])
    
    def index_from_pixel(self, pixel, column_row=False):
        # calculate the index of the row-major map
        if column_row: # pixel = [v, u]
            return int(pixel[1] * self.map_width + pixel[0])
        else: # pixel = [u, v]
            return int(pixel[0] * self.map_width + pixel[1])
        
    def pixel_from_index(self, index, column_row=False):
        # calculate the index of the row-major map
        u = int(index/self.map_width)
        v = index - u * self.map_width
        if column_row: # pixel = [v, u]
            return np.array([v,u])
        else: # pixel = [u, v]
            return np.array([u,v])

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
