#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, Quaternion

import tf_transformations
from bresenham import bresenham

import numpy as np
import math

class FastSLAM(Node):
    def __init__(self):
        super().__init__('fast_slam')
        self.get_logger().info('Inicializando FastSLAM!')

        # QoS profiles
        qos_profile_map = QoSProfile(depth=10)
        qos_profile_map.reliability = QoSReliabilityPolicy.RELIABLE
        qos_profile_map.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        
        qos_profile_scan_odom = QoSProfile(depth=10)
        qos_profile_scan_odom.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # Publishers
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', qos_profile_map)
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, '/mcl_pose', 10)
        self.pub_particles = self.create_publisher(PoseArray, '/particlecloud', 10)

        #Subscribers com sincronização
        self.sub_scan = Subscriber(self, LaserScan, "/scan", qos_profile=qos_profile_scan_odom)
        self.sub_odom = Subscriber(self, Odometry, "/odom", qos_profile=qos_profile_scan_odom)
        self.sync = ApproximateTimeSynchronizer(fs=[self.sub_scan, self.sub_odom], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.sync_callback)

        ########## Parâmetros do FastSLAM ##########
        self.dT = 0.2                                               # período do timer
        self.M = 700                                                # nº partículas 
        self.particles = np.zeros((self.M, 3), dtype=float)         # [x, y, th] para cada partícula
        self.weights = np.ones(self.M, dtype=float) / self.M        # pesos
        
        self.particle_maps = [] 

        self.current_odom = None
        self.current_scan = None
        self.previous_odom = None
        self.estimated_pose = PoseWithCovarianceStamped()
        
        self.map_resolution = 0.05
        self.map_size_meters = 20  
        self.map_size_pixels = int(self.map_size_meters / self.map_resolution)
        self.map_origin = (-self.map_size_meters/2, -self.map_size_meters/2)
        
        self.global_map = np.full((self.map_size_pixels, self.map_size_pixels), -1, dtype=np.int8)
        
        self.is_initialized = False
        self.scan_count = 0
        #############################################


    def sync_callback(self, scan_msg, odom_msg):
        orientation = odom_msg.pose.pose.orientation
        x = orientation.x
        y = orientation.y
        z = orientation.z
        w = orientation.w
        _, _, yaw = tf_transformations.euler_from_quaternion([x, y, z, w])

        position = odom_msg.pose.pose.position
        self.current_odom = (position.x, position.y, yaw)
        self.current_scan = scan_msg

        if not self.is_initialized:
            self.initialize_particles()
            self.is_initialized = True
            return

        self.fast_slam_update()
        self.publish_map()
        self.publish_particles()
        self.publish_pose()

    def initialize_particles(self):
        if self.current_odom is None:
            return

        cx, cy, cyaw = self.current_odom
        
        for i in range(self.M):
            noise_x = np.random.normal(0, 0.1)  
            noise_y = np.random.normal(0, 0.1)
            noise_yaw = np.random.normal(0, 0.05)
            
            self.particles[i, 0] = cx + noise_x
            self.particles[i, 1] = cy + noise_y
            self.particles[i, 2] = cyaw + noise_yaw
        

    def fast_slam_update(self):
        if self.previous_odom is None:
            self.previous_odom = self.current_odom
            return

        self.motion_update()
        self.map_and_correction_update()
        self.resample_particles()
        self.update_estimated_pose()
        
        self.previous_odom = self.current_odom

    def motion_update(self):
        if self.previous_odom is None:
            return

        dx = self.current_odom[0] - self.previous_odom[0]
        dy = self.current_odom[1] - self.previous_odom[1]
        dth = self.current_odom[2] - self.previous_odom[2]

        for i in range(self.M):
            noise_xy = np.random.normal(0, 0.02)  
            noise_th = np.random.normal(0, 0.01)
            
            self.particles[i, 0] += dx + noise_xy
            self.particles[i, 1] += dy + noise_xy
            self.particles[i, 2] += dth + noise_th
            
            self.particles[i, 2] = self.normalize_angle(self.particles[i, 2])

    def map_and_correction_update(self):
        if self.current_scan is None:
            return
        
        self.update_global_map()
        
        for i in range(self.M):
            particle_weight = self.particle_weight(i)
            self.weights[i] = particle_weight

        total_weight = np.sum(self.weights)
        if total_weight > 0:
            self.weights /= total_weight
        else:
            self.weights = np.ones(self.M) / self.M
            self.get_logger().warn('Todos os pesos zerados, resetando...')

    def update_global_map(self):
        """Atualiza o mapa global com as leituras atuais do laser"""
        if self.current_scan is None or self.current_odom is None:
            return

        pose = self.current_odom
        
        for j, range_measurement in enumerate(self.current_scan.ranges):
            if range_measurement < self.current_scan.range_min or range_measurement > self.current_scan.range_max:
                continue
            if math.isinf(range_measurement) or math.isnan(range_measurement):
                continue

            x0 = int((pose[0] - self.map_origin[0]) / self.map_resolution)
            y0 = int((pose[1] - self.map_origin[1]) / self.map_resolution)

            angle = pose[2] + j * self.current_scan.angle_increment + self.current_scan.angle_min
            x1 = int((pose[0] + range_measurement * math.cos(angle) - self.map_origin[0]) / self.map_resolution)
            y1 = int((pose[1] + range_measurement * math.sin(angle) - self.map_origin[1]) / self.map_resolution)

            try:
                line_cells = list(bresenham(x0, y0, x1, y1))
                
                for cell_x, cell_y in line_cells[:-1]:
                    if 0 <= cell_x < self.map_size_pixels and 0 <= cell_y < self.map_size_pixels:
                        if self.global_map[cell_y, cell_x] < 50:  # Não sobrescrever obstáculos
                            self.global_map[cell_y, cell_x] = 0
                
                if 0 <= x1 < self.map_size_pixels and 0 <= y1 < self.map_size_pixels:
                    self.global_map[y1, x1] = 100
                    
            except Exception as e:
                continue

        self.scan_count += 1

    def particle_weight(self, particle_idx):
        if self.current_scan is None:
            return 1.0 / self.M

        particle_pose = self.particles[particle_idx]
        total_weight = 0.0
        valid_rays = 0

        for j, range_measurement in enumerate(self.current_scan.ranges):
            if range_measurement < self.current_scan.range_min or range_measurement > self.current_scan.range_max:
                continue
            if math.isinf(range_measurement) or math.isnan(range_measurement):
                continue

            angle = particle_pose[2] + j * self.current_scan.angle_increment + self.current_scan.angle_min
            expected_x = particle_pose[0] + range_measurement * math.cos(angle)
            expected_y = particle_pose[1] + range_measurement * math.sin(angle)

            map_x = int((expected_x - self.map_origin[0]) / self.map_resolution)
            map_y = int((expected_y - self.map_origin[1]) / self.map_resolution)

            if 0 <= map_x < self.map_size_pixels and 0 <= map_y < self.map_size_pixels:
                map_value = self.global_map[map_y, map_x]
                if map_value == 100: 
                    total_weight += 1.0
                elif map_value == 0: 
                    total_weight += 0.1
                valid_rays += 1

        if valid_rays > 0:
            return total_weight / valid_rays
        else:
            return 1e-6

    def resample_particles(self):
        effective_particles = 1.0 / np.sum(self.weights ** 2)
        
        if effective_particles < self.M / 2:
            indices = self.systematic_resample()
            self.particles = self.particles[indices]
            self.weights = np.ones(self.M) / self.M
            
            noise_xy = np.random.normal(0, 0.01, (self.M, 2))
            noise_th = np.random.normal(0, 0.005, self.M)
            
            self.particles[:, 0:2] += noise_xy
            self.particles[:, 2] += noise_th

    def systematic_resample(self):
        positions = (np.arange(self.M) + np.random.random()) / self.M
        indices = np.zeros(self.M, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        
        while i < self.M:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        return indices

    def update_estimated_pose(self):
        mean_x = np.mean(self.particles[:, 0])
        mean_y = np.mean(self.particles[:, 1])
        
        mean_sin = np.mean(np.sin(self.particles[:, 2]))
        mean_cos = np.mean(np.cos(self.particles[:, 2]))
        mean_yaw = math.atan2(mean_sin, mean_cos)

        self.estimated_pose.header.stamp = self.get_clock().now().to_msg()
        self.estimated_pose.header.frame_id = 'map'
        self.estimated_pose.pose.pose.position.x = float(mean_x)
        self.estimated_pose.pose.pose.position.y = float(mean_y)
        
        quat = tf_transformations.quaternion_from_euler(0, 0, mean_yaw)
        self.estimated_pose.pose.pose.orientation = Quaternion(
            x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])
        )

    def publish_map(self):
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = "map"
        map_msg.info.resolution = self.map_resolution
        map_msg.info.width = self.map_size_pixels
        map_msg.info.height = self.map_size_pixels
        map_msg.info.origin.position.x = self.map_origin[0]
        map_msg.info.origin.position.y = self.map_origin[1]
        map_msg.info.origin.orientation.w = 1.0
        
        map_msg.data = self.global_map.flatten().astype(np.int8).tolist()
        
        self.map_pub.publish(map_msg)

    def publish_particles(self):
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'map'
        
        for i in range(self.M):
            pose = Pose()
            pose.position.x = float(self.particles[i, 0])
            pose.position.y = float(self.particles[i, 1])
            pose.position.z = 0.0
            
            quat = tf_transformations.quaternion_from_euler(0, 0, self.particles[i, 2])
            pose.orientation = Quaternion(
                x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])
            )
            pose_array.poses.append(pose)
        
        self.pub_particles.publish(pose_array)

    def publish_pose(self):
        self.pub_pose.publish(self.estimated_pose)

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = FastSLAM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()