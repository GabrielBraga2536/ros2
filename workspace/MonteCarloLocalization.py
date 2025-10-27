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
import random


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
        self.laser = None
        self.map = None  
        self.mcl_pose = PoseWithCovarianceStamped()
        self.declare_parameter('seed', None)
        seed = self.get_parameter('seed').value
        self.rng = np.random.default_rng(seed if seed is not None else None)
        self.initialized_particles = False
        #############################################
        # Parâmetros do modelo de movimento (odometria)
        # Alphas: ruídos (ver Thrun, Burgard, Fox - Probabilistic Robotics)
        self.alpha1 = 0.05  # ruído de rotação proporcional à rotação
        self.alpha2 = 0.05  # ruído de rotação proporcional à translação
        self.alpha3 = 0.1   # ruído de translação proporcional à translação
        self.alpha4 = 0.05  # ruído de translação proporcional à rotação

        # Parâmetros do modelo do sensor (Laser)
        self.z_sigma = 0.25         # desvio padrão da medição esperada (m)
        self.max_beams = 60         # número máximo de feixes a considerar (downsample)
        self.max_range = None        # será definido a partir da mensagem de laser

        # Cache do mapa em numpy
        self.map_grid = None         # np.ndarray (H, W) com ocupação (0 livre, 1 ocupado, -1 desconhecido)
        self.map_info = None         # dict com resolution, width, height, origin (x,y,theta)

        # Odometria anterior para delta
        self.last_odom = None

        self.declare_parameter('particles', self.M)
        p_value = self.get_parameter('particles').value

        self.timer = self.create_timer(self.dT, self.timer_callback)
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
        if self.max_range is None:
            # guarda o alcance máximo do laser para raycasting e limites
            self.max_range = float(msg.range_max)

    def map_callback(self, msg):
        self.map = msg
        if not self.initialized_particles:
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
        # Garante que há mapa
        if self.map is None:
            return

        # Constrói grid e infos a partir do OccupancyGrid
        info = self.map.info
        self.width = int(info.width)
        self.height = int(info.height)
        self.res = float(info.resolution)
        self.origin_x = float(info.origin.position.x)
        self.origin_y = float(info.origin.position.y)

        data = np.array(self.map.data, dtype=np.int16).reshape((self.height, self.width))

        self.occup = (data >= 65)

        livres = np.argwhere((data >= 0) & (data <= 50))

        if (livres.size == 0):
            self.P[:, 0:2] = 0.0
            self.P[:, 2] = np.random.uniform(-math.pi, math.pi, size = self.M)
        else:
            index = np.random.choice(livres.shape[0], size = self.M, replace = True)
            row_column = livres[index]
            xs = self.origin_x + (row_column[:, 1].astype(float) + 0.5) * self.res
            ys = self.origin_y + (row_column[:, 0].astype(float) + 0.5) * self.res
            theta = np.random.uniform(-math.pi, math.pi, size = self.M)
            self.p[:, 0] = xs
            self.p[:, 1] = ys
            self.p[:, 2] = theta

        self.w[:] = 1.0 / self.M
        
        self.last_odom = np.array(self.odom if self.odom is not None else [0.0, 0.0, 0.0], dtype = float)

    # algorithm - atualização
    def mcl_algorithm(self):
        # Garante que há odometria, laser e mapa
        if self.odom is None or self.laser is None or self.map_grid is None:
            return

        # Etapa 1: Previsão 
        current = np.array(self.odom, dtype = float)
        d = current - self.last_odom
        self.last_odom = current.copy()

        # Ruído nas componentes
        self.p[:, 0] += delta[0] + np.random.normal(0.0, 0.02, self.M)
        self.p[:, 1] += delta[1] + np.random.normal(0.0, 0.02, self.M)
        self.p[:, 2] += delta[2] + np.random.normal(0.0, 0.01, self.M)
        self.p[:, 2] = (self.p[:, 2] + math.pi) % (2 * math.pi) - math.pi

        # Etapa 2: Correção (atualização por sensor)
        # Modelo de sensor: verossimilhança Gaussiana com distância esperada via raycast no mapa
        laser = self.laser
        ranges = np.array(laser.ranges, dtype = float)
        angles = np.linspace(laser.angle_min, laser.angle_max, len(ranges), dtype = float)
        step = max(1, len(ranges) // 30)
        weights = np.zeros(self.M, dtype = float)

        for i in range(self.M):
            x, y, theta = self.p[i]
            score = 0

            for a, r in zip(angles[::step], ranges[::step]):

                if not np.isfinite(r):
                    continue
                
                ex = x + r * math.cos(theta + a)
                ey = y + r * math.sin(theta + a)
                col = int((ex - self.origin_x) / self.res)
                row = int((ey - self.origin_y) / self.res)
                
                if 0 <= row < self.height and 0 <= col < self.width:
                    if self.occup[row, col]:
                        score += 1
                weights[i] = score + 1e-6
        
        soma = float(np.sum(pesos))

        if (soma > 0.0 and np.isfinite(soma)):
            self.w = weights / soma
        else:
            self.w[:] = 1.0 / self.M


        # Etapa 3: Reamostragem (low-variance)
        ef = 1.0 / np.sum(self.w ** 2)

        if (ef < self.M * 0.5):
            index = np.random.choice(self.M, self.M, p = self.w)
            self.w[:] = 1.0 / self.M

        # Etapa 4: Estimativa da posição (média ponderada)
        mx = float(np.mean(self.p[:, 0]))
        my = float(np.mean(self.p[:, 1]))
        mth = float(np.mean(self.p[:, 2]))
        print(f"Estimativa: x = {mx:.2f}, y = {my:.2f}, theta = {mth:.2f}")

        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0.0, 0.0, mth)
        self.mcl_pose.header.frame_id = "map"
        self.mcl_pose.header.stamp = self.get_clock().now().to_msg()
        self.mcl_pose.pose.pose.position.x = mx
        self.mcl_pose.pose.pose.position.y = my
        self.mcl_pose.pose.pose.position.z = 0.0
        self.mcl_pose.pose.pose.orientation.x = float(qx)
        self.mcl_pose.pose.pose.orientation.y = float(qy)
        self.mcl_pose.pose.pose.orientation.z = float(qz)
        self.mcl_pose.pose.pose.orientation.w = float(qw)

        self.particlecloud.header.frame_id = "map"
        self.particlecloud.header.stamp = self.get_clock().now().to_msg()
        self.particlecloud.poses.clear()

        for i in range(self.M):
            pose = Pose()
            pose.position.x = float(self.p[i, 0])
            pose.position.y = float(self.p[i, 1])
            pose.position.z = 0.0
            qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0.0, 0.0, float(self.p[i, 2]))
            pose.orientation.x = float(qx)
            pose.orientation.y = float(qy)
            pose.orientation.z = float(qz)
            pose.orientation.w = float(qw)
            self.particlecloud.poses.append(pose)

        # Publica
        self.pose_pub.publish(self.mcl_pose)
        self.pcloud_pub.publish(self.particlecloud)


def main(args=None):
    rclpy.init(args=args) # Inicializando ROS
    node = MCL()          # Inicializando nó
    del node              # Finalizando nó
    rclpy.shutdown()      # Finalizando ROS

if __name__ == '__main__':
    main()
