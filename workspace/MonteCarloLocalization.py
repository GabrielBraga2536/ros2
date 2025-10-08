#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, Quaternion

import tf_transformations

import numpy as np
import math

class MCL(Node):
    def __init__(self):
        super().__init__('mcl')
        self.get_logger().info('Inicializando o nó!')

        qos_profile_map = QoSProfile(depth=10)
        qos_profile_map.reliability = QoSReliabilityPolicy.RELIABLE
        qos_profile_map.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        qos_profile_scan_odom = QoSProfile(depth=10)
        qos_profile_scan_odom.reliability = QoSReliabilityPolicy.BEST_EFFORT

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_profile_map)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_scan_odom)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, qos_profile_scan_odom)
    
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, '/mcl_pose', 10)
        self.pub_particles = self.create_publisher(PoseArray, '/particlecloud', 10)

        ########## inicializando variáveis ##########
        self.dT = 0.2                                               # período do timer
        self.M = 1000                                               # nº partículas
        self.p = np.zeros((self.M, 3), dtype=float)                 # [x, y, th]
        self.w = np.ones(self.M, dtype=float) / self.M              # pesos
        self.odom = None
        self.scan = None
        self.map = None  
        self.mcl_pose = PoseWithCovarianceStamped()
        #############################################

        self.timer = self.create_timer(self.dT,self.timer_callback)
        rclpy.spin(self)

    # Finalizando nó
    def __del__(self):
        self.get_logger().info('Finalizando o nó!')
        self.destroy_node()

    def odom_callback(self, msg):
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        _, _, yaw = tf_transformations.euler_from_quaternion([x, y, z, w])

        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y
        self.odom = (pos_x, pos_y, yaw)

    def laser_callback(self, msg):
        self.laser = msg

    def map_callback(self, msg):
        self.map = msg

        self.inicializacao()

    # Executando
    def timer_callback(self):
        if self.odom is None or self.laser == None or self.map == None:
            return       
        
        self.mcl_algorithm()
        self.publicar_pose()
        self.publicar_particulas()

    def publicar_pose(self):
        msg = self.mcl_pose 
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'     
        self.pub_pose.publish(msg)
        
    def publicar_particulas(self):
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = 'map'
        for i in range(self.M):
            x, y, th = self.p[i]
            q = tf_transformations.quaternion_from_euler(0.0, 0.0, th)
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
            pa.poses.append(pose)
        self.pub_particles.publish(pa)

    # algorithm - inicialização
    def inicializacao(self):
        pass

    # algorithm - atualização
    def mcl_algorithm(self):
        # Previsão
        # Correção
        # Reamostragem
        # Estimativa da posição
        pass


def main(args=None):
    rclpy.init(args=args) # Inicializando ROS
    node = MCL()          # Inicializando nó
    del node              # Finalizando nó
    rclpy.shutdown()      # Finalizando ROS

if __name__ == '__main__':
    main()
