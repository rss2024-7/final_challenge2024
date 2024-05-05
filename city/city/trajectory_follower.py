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
        self.declare_parameter('drive_topic', "default")

        self.min_dist = 0
        self.car_pos = np.array([0, 0])
        self.car_angle = 0

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.dist_from_shell = 0.5
        self.follow_shell = False

        self.lane_offset = 0.4


        self.lookahead = 1  # FILL IN #
        self.speed = 4.0  # FILL IN #
        self.wheelbase_length = 0.3  # FILL IN #

        # Adjust lookahead based on angle to lookahead point
        # higher angle error ~ lower lookahead distance
        self.min_lookahead = 1.0
        self.max_lookahead = 2.0 

        self.speed_to_lookahead = 1.0
        
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
        
        self.target_pub = self.create_publisher(Marker, "/target_point", 1)
        self.radius_pub = self.create_publisher(Marker, "/radius", 1)

        self.turn_success_pub = self.create_publisher(Bool, "/turn_outcome", 1)

        self.turn_sub = self.create_subscription(Float32,
                                            "/turnaround",
                                            self.turnaround_callback,
                                            1)


        self.error_pub = self.create_publisher(Float32, "/error", 1)
        self.num_dist = 0
        self.tot_dist = 0

        self.initialized_traj = False

        self.declare_parameter("lane", "default")
        path = self.get_parameter("lane").get_parameter_value().string_value
        self.lane_traj = LineTrajectory(self, "/lane")
        self.lane_traj.load(path)

        # need to wait a short period of time before publishing the first message
        time.sleep(0.5)

        # visualize the loaded trajectory
        self.lane_traj.publish_viz()

        self.shell_pub = self.create_publisher(Marker, "/shell_point", 3)
        self.shell_near_pub = self.create_publisher(Marker, "/shell_near_point", 3)

        self.shell_traj = LineTrajectory(self, "/shellpath")
        self.shell_locations = []

        self.num_shells = 0
        
        self.car_in_endzone = True

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
    
    def in_bounds(self, lower_bounds, upper_bounds, x, y):
        x_in = (lower_bounds[:, 0] <= x) & (x <= upper_bounds[:, 0])
        y_in = (lower_bounds[:, 1] <= y) & (y <= upper_bounds[:, 1])

        return np.any(x_in & y_in)




    def point_callback(self, point_msg):
        # self.initialized_traj = True
        # near_point = self.find_closest_point(point_msg.point.x, point_msg.point.y)

        # self.get_logger().info(f'{point_msg.point.x=} {point_msg.point.y=}')
        # self.get_logger().info(f'{near_point=}')

        # self.pub_point(self.shell_pub, (0.0, 1.0, 0.0), (point_msg.point.x, point_msg.point.y))
        # self.pub_point(self.shell_near_pub, (1.0, 0.0, 0.0), near_point)

        self.num_shells += 1

    def turnaround_callback(self, msg):
        if self.follow_shell:
            turn_msg = Bool()
            turn_msg.data = False
            self.turn_success_pub.publish(turn_msg) 
            return
        self.lane_traj.points.reverse()

        # 3-Point Turn
        drive_msg = AckermannDriveStamped()

        drive_msg.drive.speed = 1.0
        drive_msg.drive.steering_angle = self.max_steer
        self.drive_pub.publish(drive_msg)
        time.sleep(1.25)

        drive_msg.drive.speed = -1.0
        drive_msg.drive.steering_angle = -self.max_steer
        self.drive_pub.publish(drive_msg)
        time.sleep(0.75)

        drive_msg.drive.speed = 1.0
        drive_msg.drive.steering_angle = self.max_steer
        self.drive_pub.publish(drive_msg)
        time.sleep(.75)

        turn_msg = Bool()
        turn_msg.data = True
        self.turn_success_pub.publish(turn_msg)



    def pose_callback(self, odometry_msg):

        # self.get_logger().info(f"{self.num_shells=}")

        if self.num_shells < 3: return

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

        # # IN END ZONE, U TURN
        # car_in_endzone = self.in_endzone(car_pos_x, car_pos_y)
        # if car_in_endzone and not self.car_in_endzone:
        #     self.lane_traj.points.reverse()

        #     # # U - TURN
        #     # drive_msg = AckermannDriveStamped()
        #     # drive_msg.drive.speed = 1.0
        #     # steer_angle = self.max_steer
        #     # drive_msg.drive.steering_angle = steer_angle
        #     # self.drive_pub.publish(drive_msg)
        #     # time.sleep(2)


        #     # 3-Point Turn
        #     drive_msg = AckermannDriveStamped()

        #     drive_msg.drive.speed = 1.0
        #     drive_msg.drive.steering_angle = self.max_steer
        #     self.drive_pub.publish(drive_msg)
        #     time.sleep(1.25)

        #     drive_msg.drive.speed = -1.0
        #     drive_msg.drive.steering_angle = -self.max_steer
        #     self.drive_pub.publish(drive_msg)
        #     time.sleep(0.75)

        #     drive_msg.drive.speed = 1.0
        #     drive_msg.drive.steering_angle = self.max_steer
        #     self.drive_pub.publish(drive_msg)
        #     time.sleep(.75)

        #     self.car_in_endzone = True
        #     return
        # if not car_in_endzone and self.car_in_endzone:
        #     self.car_in_endzone = False

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
            car_to_seg_start_x < 0 and \
            -2 < car_to_seg_end_x < 0 and \
                car_to_seg_end_x ** 2 + car_to_seg_end_y ** 2 <= 2): 
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            self.drive_pub.publish(drive_msg)

            if self.follow_shell:
                time.sleep(5)
                drive_msg = AckermannDriveStamped()
                drive_msg.drive.speed = -1.0
                self.drive_pub.publish(drive_msg)
                # time.sleep(2)
                time.sleep(np.linalg.norm(np.array(self.shell_traj.points[0]) - np.array(self.shell_traj.points[1])))
                self.follow_shell = False
                return
            
            
            # avg_dist = self.tot_dist / self.num_dist
            # self.get_logger().info("Average error: %s" % avg_dist)
            # self.initialized_traj = False
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


        if not self.follow_shell:
            car_to_target_x -= self.lane_offset
            car_to_target_y -= self.lane_offset


        # # U TURN
        # if closest_segment_index + 1 == len(traj_x) - 1 and car_to_target_x < -0.5 and not self.follow_shell:
        #     drive_msg = AckermannDriveStamped()
        #     drive_msg.drive.speed = self.lookahead * self.speed_to_lookahead
        #     steer_angle = self.max_steer
        #     drive_msg.drive.steering_angle = steer_angle
        #     self.drive_pub.publish(drive_msg)
        #     return

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

        steer_angle = np.arctan2((self.wheelbase_length*np.sin(angle_error)), 
                                 0.5*self.lookahead + self.wheelbase_length*np.cos(angle_error))
        
        steer_angle = np.clip(steer_angle, -self.max_steer, self.max_steer)
        drive_msg.drive.steering_angle = steer_angle
        self.drive_pub.publish(drive_msg)

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

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

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = -1.0
        drive_msg.drive.steering_angle = np.sign(angle) * self.max_steer
        self.drive_pub.publish(drive_msg)
        time.sleep(self.wheelbase_length * np.abs(angle) / np.tan(self.max_steer))

        # self.trajectory.save("current_trajectory.traj")

        self.follow_shell = True

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