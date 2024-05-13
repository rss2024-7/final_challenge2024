import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray, PointStamped, PoseWithCovarianceStamped, Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32, ColorRGBA, Bool
from wall_follower.visualization_tools import VisualizationTools
import numpy as np

from .utils import LineTrajectory
import time


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "/vesc/input/navigation")

        self.min_dist = 0
        self.car_pos = np.array([0, 0])
        self.car_angle = 0
        self.real_car_angle_offset = - 0.03675

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.get_logger().info(f"{self.drive_topic=}")

        self.dist_from_shell = 0.5
        self.follow_shell = False

        self.lane_offset = 0.5


        self.lookahead = 1  # FILL IN #
        self.speed = 4.0  # FILL IN #
        self.wheelbase_length = 0.3  # FILL IN #

        # Adjust lookahead based on angle to lookahead point
        # higher angle error ~ lower lookahead distance
        self.min_lookahead = 1.0
        self.max_lookahead = 2.0 

        self.speed_to_lookahead = 0.7
        # self.speed_to_lookahead = 0.5
        
        # the angle to target s.t. the lookahead will be at its minimum
        self.min_lookahead_angle = np.deg2rad(90) 
        
        self.max_steer = 0.34

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        
        self.pose_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.pose_callback,
                                                 1)
        
        self.point_sub = self.create_subscription(PointStamped,
                                                 "/clicked_point",
                                                 self.point_callback,
                                                 1)
        self.init_point_sub = self.create_subscription(PoseWithCovarianceStamped,
                                            "/initialpose",
                                            self.init_callback,
                                            1)
        self.stopsign_sub = self.create_subscription(Bool,
                                        "/stopsign",
                                        self.stopsign_callback,
                                        1)
        self.traffic_sub = self.create_subscription(Bool,
                                "/light_detector",
                                self.green_callback,
                                1)
        
        self.target_pub = self.create_publisher(Marker, "/target_point", 1)
        self.radius_pub = self.create_publisher(Marker, "/radius", 1)

        self.turn_success_pub = self.create_publisher(Bool, "/turn_outcome", 1)

        self.traffic_zone_pub = self.create_publisher(Bool, "/traffic_zone", 1)

        self.turn_sub = self.create_subscription(Float32,
                                            "/turnaround",
                                            self.turnaround_callback,
                                            1)


        self.error_pub = self.create_publisher(Float32, "/error", 1)
        self.num_dist = 0
        self.tot_dist = 0

        self.initialized_traj = False

        self.traffic_timer = 0

        self.declare_parameter("lane", "default")
        path = self.get_parameter("lane").get_parameter_value().string_value
        self.lane_traj = LineTrajectory(self, "/lane")
        self.lane_traj.load(path)

        # need to wait a short period of time before publishing the first message
        time.sleep(0.5)

        # visualize the loaded trajectory
        self.lane_traj.publish_viz()

        self.shell_collect_pub = self.create_publisher(Bool, "/shell_collected", 3)

        self.shell_traj = LineTrajectory(self, "/shellpath")
        self.shell_locations = []

        self.num_shells = 0
        
        self.car_in_endzone = True

        self.reversing = False
        self.reverse_start = None
        self.reverse_time = None # seconds

        self.turn_stage = 0
        self.turn_stage_start = 0
        self.turn_time = 0

        self.parked = False
        self.park_start = None
        self.park_time = None 

        self.hard_steering = False

        self.pillar_zone_lower = np.array([[-19, 10]])
        self.pillar_zone_upper = np.array([[-15, 11]])

        # self.lane_traj.points.reverse()

        self.traffic_locations = np.array([[-10.5, 16.6],
                                            [-28.7, 34.1],
                                            [-54.8, 24.6]])
        self.traffic_zone_radius = 1.3 # 1.5 # meters
        self.traffic_buffer_radius = 3 #3.05
        
        self.is_near_traffic = False
        self.traffic_stop = False

        self.green_lights = False
        self.is_in_buffer_zone = False
        self.is_in_traffic_buffer_zone = False

        self.traffic_cooldown = 0

        self.go_timer = 0

        self.jitter_count = 0
        self.jitter_timer = 0

        self.get_logger().info("Follower Ready") 
    
    def jitter_wheels(self):
        if time.time() - self.jitter_timer > .5: 
            self.jitter_count += 1
            self.jitter_timer = time.time()

        angle = self.max_steer
        if self.jitter_count % 2 == 0: angle = - self.max_steer

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 0.0
        drive_msg.drive.steering_angle = angle + self.real_car_angle_offset
        self.drive_pub.publish(drive_msg)


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

        projection_proportion = np.dot(end - start, p - start) / np.linalg.norm(end - start) ** 2
        closest_point = start + projection_proportion * (end - start)

        dist = np.linalg.norm(closest_point - p)
        
        near_point = p + self.dist_from_shell / dist * (closest_point - p)

        return near_point
    
    def in_buffer_zone(self, x, y):

        pos = np.array([x, y])
        delta = self.traffic_locations - pos
        dists = np.linalg.norm(delta, axis=1)

        close = (self.traffic_zone_radius <= dists) & (dists <= self.traffic_buffer_radius)

        return np.any(close)
    
    def in_traffic_and_buffer_zone(self, x, y):

        pos = np.array([x, y])
        delta = self.traffic_locations - pos
        dists = np.linalg.norm(delta, axis=1)

        close = dists <= self.traffic_buffer_radius

        in_zone = np.any(close)

        zone_msg = Bool()
        if in_zone:
            zone_msg.data = True
        else:
            zone_msg.data = False
        self.traffic_zone_pub.publish(zone_msg)

        return in_zone

    
    def in_bounds(self, lower_bounds, upper_bounds, x, y):
        x_in = (lower_bounds[:, 0] <= x) & (x <= upper_bounds[:, 0])
        y_in = (lower_bounds[:, 1] <= y) & (y <= upper_bounds[:, 1])

        return np.any(x_in & y_in)

    def in_pillar_zone(self, x, y):
        return self.in_bounds(self.pillar_zone_lower, self.pillar_zone_upper, x, y)
    
    def green_callback(self, msg):
        if self.is_in_traffic_buffer_zone:
            self.green_lights = self.green_lights or msg.data



    def point_callback(self, point_msg):
        # self.initialized_traj = True
        # near_point = self.find_closest_point(point_msg.point.x, point_msg.point.y)

        # self.get_logger().info(f'{point_msg.point.x=} {point_msg.point.y=}')
        # self.get_logger().info(f'{near_point=}')

        # self.pub_point(self.shell_pub, (0.0, 1.0, 0.0), (point_msg.point.x, point_msg.point.y))
        # self.pub_point(self.shell_near_pub, (1.0, 0.0, 0.0), near_point)

        self.num_shells += 1

    def reverse(self, reverse_time, angle=0.0):
        self.hard_steering = True
        time_start = time.time()
        while time.time() - time_start < reverse_time + 0.2:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = -1.0
            drive_msg.drive.steering_angle = angle + self.real_car_angle_offset
            self.drive_pub.publish(drive_msg)
        
        green_msg = Bool()
        green_msg.data = True
        self.green_callback(green_msg)
        self.is_in_traffic_buffer_zone = False
        self.is_in_buffer_zone = False
        self.traffic_cooldown = 0
        self.hard_steering = False

    def park(self, park_time):
        self.hard_steering = True
        time_start = time.time()
        while time.time() - time_start < park_time:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            self.drive_pub.publish(drive_msg)
        self.hard_steering = False

    def stopsign_callback(self, msg):
        self.get_logger().info("stopping for stop sign...")
        self.park(2)

    def turnaround_callback(self, msg):
        self.hard_steering = True
        if self.follow_shell:
            turn_msg = Bool()
            turn_msg.data = False
            self.turn_success_pub.publish(turn_msg) 
            return

        # Forward Left
        time_start = time.time()
        while time.time() - time_start < 1.0:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 1.0
            drive_msg.drive.steering_angle = self.max_steer
            self.drive_pub.publish(drive_msg)

        # Back Right
        time_start = time.time()
        while time.time() - time_start < 1.25:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = -1.0
            drive_msg.drive.steering_angle = -self.max_steer + self.real_car_angle_offset
            self.drive_pub.publish(drive_msg)

        # Forward Left
        time_start = time.time()
        while time.time() - time_start < 0.0:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 1.0
            drive_msg.drive.steering_angle = self.max_steer / 2
            self.drive_pub.publish(drive_msg)

        self.lane_traj.points.reverse()

        self.is_in_traffic_and_buffer_zone = False
        self.is_in_buffer_zone = False
        self.traffic_cooldown = time.time()
        
        turn_msg = Bool()
        turn_msg.data = True
        self.turn_success_pub.publish(turn_msg)
        self.hard_steering = False




    def pose_callback(self, odometry_msg):
        # self.get_logger().info("receiving pose. jittering...")
        if self.hard_steering: return
        if self.traffic_stop: 
            # self.get_logger().info("stopping...")
            if time.time() - self.go_timer < 10 and not self.green_lights: 
                self.jitter_wheels()
                return
            if time.time() - self.go_timer > 10:
                self.get_logger().info("TOO LONG OF A WAIT. GO GO GO")
            if self.green_lights:
                self.get_logger().info("GREEN LIGHT GO GO GO")
            self.get_logger().info("GREEN GO GO GO")
            self.traffic_cooldown = time.time()
            self.traffic_stop = False
      
        # if self.num_shells < 3: return
        # process trajectory points into np arrays
        points = np.array(self.lane_traj.points)

        # shell path to follow
        if self.follow_shell: 
            points = np.array(self.shell_traj.points)
    
        traj_x = points[:, 0]
        traj_y = points[:, 1]

        # retrieve odometry data
        car_pos_x = odometry_msg.pose.pose.position.x
        car_pos_y = odometry_msg.pose.pose.position.y
        angle = 2 * np.arctan2(odometry_msg.pose.pose.orientation.z, odometry_msg.pose.pose.orientation.w)

        self.is_in_traffic_buffer_zone = self.in_traffic_and_buffer_zone(car_pos_x, car_pos_y)

        # stop near traffic light
        if time.time() - self.traffic_cooldown > 10 and self.in_buffer_zone(car_pos_x, car_pos_y):
            if not self.is_in_buffer_zone:
                self.get_logger().info("Entering Buffer")
                self.is_in_buffer_zone = True
                self.green_lights = False
                zone_msg = Bool()
                zone_msg.data = True
                # self.traffic_zone_pub.publish(zone_msg)
        else:
            if self.is_in_buffer_zone:
                self.get_logger().info("Exiting Buffer")
                zone_msg = Bool()
                zone_msg.data = False

                self.is_in_buffer_zone = False
                
                # self.traffic_zone_pub.publish(zone_msg)
                if not self.green_lights:
                    self.traffic_stop = True
                    self.get_logger().info("RED LIGHT STOP")
                    self.go_timer = time.time()
                    return
                self.get_logger().info("GREEN GO")
                self.traffic_cooldown = time.time()
            self.is_in_buffer_zone = False

        car_angle = 2 * np.arctan2(odometry_msg.pose.pose.orientation.z, odometry_msg.pose.pose.orientation.w)

        self.car_pos = np.array([car_pos_x, car_pos_y])
        self.car_angle = car_angle 

        # get info about closest segment
        closest_segment_index = self.find_closest_segment(traj_x, traj_y, car_pos_x, car_pos_y)
        seg_start_x, seg_start_y = traj_x[closest_segment_index], traj_y[closest_segment_index]
        car_to_seg_start_x, car_to_seg_start_y = self.to_car_frame(seg_start_x, seg_start_y, car_pos_x, car_pos_y, car_angle)
        seg_end_x, seg_end_y = traj_x[closest_segment_index + 1], traj_y[closest_segment_index + 1]
        car_to_seg_end_x, car_to_seg_end_y = self.to_car_frame(seg_end_x, seg_end_y, car_pos_x, car_pos_y, car_angle)

        # on last segment and past end
        if (closest_segment_index + 1 == len(traj_x) - 1 and \
            car_to_seg_start_x < 2 and \
            -2 < car_to_seg_end_x < 2 and \
                car_to_seg_end_x ** 2 + car_to_seg_end_y ** 2 <= 1): 
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            self.drive_pub.publish(drive_msg)

            if self.follow_shell:
                self.park(5)

                green_msg = Bool()
                green_msg.data = True
                self.green_callback(green_msg)
                if self.in_traffic_and_buffer_zone(car_pos_x, car_pos_y):
                    self.traffic_cooldown = time.time()

                self.follow_shell = False
                msg = Bool()
                msg.data = True
                self.shell_collect_pub.publish(msg)
                return

            return


        lookahead_point = self.get_lookahead_point(traj_x[closest_segment_index], traj_y[closest_segment_index],
                                                    traj_x[closest_segment_index + 1], traj_y[closest_segment_index + 1],
                                                    car_pos_x, car_pos_y)

        # default target point is end of closest segment (in case no lookahead point found)
        target_point_x, target_point_y = seg_end_x, seg_end_y

        # if found lookahead point, set target to it
        if lookahead_point is not None:
            target_point_x, target_point_y = lookahead_point

        # convert target point to the car's frame
        car_to_target_x, car_to_target_y = self.to_car_frame(target_point_x, target_point_y, car_pos_x, car_pos_y, car_angle)

        # apply lane offset
        if not self.follow_shell and not self.in_pillar_zone(target_point_x, target_point_y): # and not self.in_pillar_zone(car_pos_x, car_pos_y) and not self.in_pillar_zone(car_to_target_x, car_to_seg_start_y):
            # car_to_target_y -= self.lane_offset

            angle = np.arctan2(car_to_target_y, car_to_target_x)

            if abs(angle) <= np.pi / 6:
                car_to_target_y -= self.lane_offset
            else:
                car_to_target_y -= np.sign(angle) * self.lane_offset / 4
                car_to_target_x += self.lane_offset

            # if abs(angle) >= np.pi/12:
            #     car_to_target_x += np.sign(angle) * self.lane_offset * .25

        if self.in_pillar_zone(target_point_x, target_point_y): # self.in_pillar_zone(car_pos_x, car_pos_y) or self.in_pillar_zone(car_to_target_x, car_to_seg_start_y):
            car_to_target_y += self.lane_offset

        # Visualize Stuff
        VisualizationTools.plot_line(np.array([0, car_to_target_x]), np.array([0, car_to_target_y]), self.target_pub, frame="base_link")
        angles = np.linspace(-np.pi, np.pi, 100)
        circle_x = self.lookahead * np.cos(angles)
        circle_y = self.lookahead * np.sin(angles)
        VisualizationTools.plot_line(circle_x, circle_y, self.radius_pub, frame="base_link")

        # angle to target point
        angle_error = np.arctan2(car_to_target_y, car_to_target_x)

        self.lookahead = self.max_lookahead \
                            - np.clip(np.abs(angle_error), 0, 
                                        self.min_lookahead_angle) / self.min_lookahead_angle \
                                        * (self.max_lookahead - self.min_lookahead)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.lookahead * self.speed_to_lookahead 
        if self.is_in_buffer_zone: drive_msg.drive.speed = 0.5 

        steer_angle = np.arctan2((self.wheelbase_length*np.sin(angle_error)), 
                                 0.5*self.lookahead + self.wheelbase_length*np.cos(angle_error))
        
        steer_angle = np.clip(steer_angle, -self.max_steer, self.max_steer)
        drive_msg.drive.steering_angle = steer_angle + self.real_car_angle_offset
        self.drive_pub.publish(drive_msg)


    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")
        self.follow_shell = True

        self.shell_traj.clear()
        self.shell_traj.fromPoseArray(msg)
        self.shell_traj.publish_viz(duration=0.0)

        seg_start = self.shell_traj.points[0]
        seg_end = self.shell_traj.points[1]

        car_to_start = self.to_car_frame(seg_start[0], seg_start[1], self.car_pos[0], self.car_pos[1], self.car_angle)
        car_to_end = self.to_car_frame(seg_end[0], seg_end[1], self.car_pos[0], self.car_pos[1], self.car_angle)

        diff = car_to_end - car_to_start
        slope = diff[1] / diff[0]
        angle = -np.arctan(slope)

        self.park(0.1)
        self.reverse(self.wheelbase_length * np.abs(angle) / np.tan(self.max_steer), np.sign(angle) * self.max_steer)


        

    def find_closest_segment(self, x, y, car_x, car_y):
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
        points = np.vstack((x, y)).T
        v = points[:-1, :] # segment start points
        w = points[1:, :] # segment end points
        p = np.array([[car_x, car_y]])
        
        l2 = np.sum((w - v)**2, axis=1)

        l2 += 10e-10

        t = np.maximum(0, np.minimum(1, np.sum((p - v) * (w - v), axis=1) / l2))

        projections = v + t[:, np.newaxis] * (w - v)
        min_distances = np.linalg.norm(p - projections, axis=1)

        min_dist = np.min(min_distances)
        self.min_dist = min_dist
        error_msg = Float32()
        error_msg.data = min_dist
        self.error_pub.publish(error_msg)

        self.num_dist += 1
        self.tot_dist += min_dist

        # if too close to end point of segment, take it out of consideration for closest line segment
        end_point_distances = np.linalg.norm(w-p, axis=1)
        min_distances[np.where(end_point_distances[:-1] < self.lookahead)] += 500

        if np.min(min_distances) > self.lookahead:
            self.lookahead = np.min(min_distances)

        closest_segment_index = np.where(min_distances == np.min(min_distances))

        if len(closest_segment_index) == 0: return 0
        if len(closest_segment_index[0]) == 0: return 0

        return closest_segment_index[0][0]
    



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
        r = self.lookahead  # Radius of circle

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

        t = max(t1, t2)

        if (t < 0): return None

        return P1 + t * V
    
    # convert a point from the map frame to the car frame
    def to_car_frame(self, x, y, car_x, car_y, car_angle):
        def rotation_matrix(angle):
            return np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
    
        # convert [x y theta] to 3x3 transform matrix
        def transform_matrix(pose):
            X = np.eye(3)
            X[:2, :2] = rotation_matrix(pose[2])
            X[:2, -1] = np.array(pose[:2])
            return X
        world_to_car = transform_matrix([car_x, car_y, car_angle])
        world_to_target = transform_matrix([x, y, 0.0])
        car_to_target = np.linalg.inv(world_to_car) @ world_to_target

        return car_to_target[:2, 2]
    
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

    def init_callback(self, msg):
        self.num_dist = 0
        self.tot_dist = 0
        # self.initialized_traj = False


def main(args=None):



    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
