#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan

import tf_transformations
from bresenham import bresenham

import numpy as np
import math

class Mapeamento(Node):
    def __init__(self):
        super().__init__('mapeamento')
        self.get_logger().info('Inicializando o nó!')

        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = QoSReliabilityPolicy.RELIABLE
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        self.map_pub = self.create_publisher(OccupancyGrid, '/map', qos_profile)
        self.sub_scan = Subscriber(self, LaserScan, "/scan")
        self.sub_odom = Subscriber(self, Odometry, "/odom")
        self.sync = ApproximateTimeSynchronizer(fs=[self.sub_scan, self.sub_odom], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.sync_callback)

        # inicializando variáveis
        self.pose = None
        self.map_res = 0.05
        self.map_real_size = 20 # mXm
        self.map_size = int(self.map_real_size / self.map_res)
        self.map = np.full((self.map_size, self.map_size), -1, dtype=np.int8)  
        self.origin = (-self.map_real_size/2, -self.map_real_size/2) # centro do mapa

        rclpy.spin(self)

    # Finalizando nó
    def __del__(self):
        self.get_logger().info('Finalizando o nó!')
        self.destroy_node()

    # Executando
    def sync_callback(self, msg_scan, msg_odom):
        x = msg_odom.pose.pose.orientation.x
        y = msg_odom.pose.pose.orientation.y
        z = msg_odom.pose.pose.orientation.z
        w = msg_odom.pose.pose.orientation.w
        _, _, yaw = tf_transformations.euler_from_quaternion([x, y, z, w])

        pos_x = msg_odom.pose.pose.position.x
        pos_y = msg_odom.pose.pose.position.y
        self.pose = (pos_x, pos_y, yaw)

        self.laser = msg_scan

        self.mapping_algorithm()
        self.publicar_mapa()

    def mapping_algorithm(self):
        pos_x, pos_y, yaw = self.pose
        ranges = self.laser.ranges
        angle_min = self.laser.angle_min
        angle_increment = self.laser.angle_increment
        max_range = self.laser.range_max

        # Posição do robô no mapa
        map_x = int((pos_x - self.origin[0]) / self.map_res)
        map_y = int((pos_y - self.origin[1]) / self.map_res)

        for i, r in enumerate(ranges):
            if np.isinf(r) or np.isnan(r) or r > max_range:
                continue

            angle = angle_min + i * angle_increment + yaw
            # Coordenada do ponto de impacto do laser
            end_x = pos_x + r * math.cos(angle)
            end_y = pos_y + r * math.sin(angle)
            map_end_x = int((end_x - self.origin[0]) / self.map_res)
            map_end_y = int((end_y - self.origin[1]) / self.map_res)

            # Traça linha do robô até o obstáculo (livre)
            for cell in bresenham(map_x, map_y, map_end_x, map_end_y):
                x, y = cell
                if 0 <= x < self.map_size and 0 <= y < self.map_size:
                    self.map[y, x] = 0  # livre

            # Marca obstáculo
            if 0 <= map_end_x < self.map_size and 0 <= map_end_y < self.map_size:
                self.map[map_end_y, map_end_x] = 100 # ocupado


    def publicar_mapa(self):
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.map_res
        msg.info.width = self.map_size
        msg.info.height = self.map_size
        msg.info.origin.position.x = self.origin[0]
        msg.info.origin.position.y = self.origin[1]
        msg.data = self.map.flatten().tolist()
        self.map_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args) # Inicializando ROS
    node = Mapeamento()   # Inicializando nó
    del node              # Finalizando nó
    rclpy.shutdown()      # Finalizando ROS

if __name__ == '__main__':
    main()
