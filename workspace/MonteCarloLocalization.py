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
        self.laser = None
        self.map = None  
        self.mcl_pose = PoseWithCovarianceStamped()
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
        width = int(info.width)
        height = int(info.height)
        res = float(info.resolution)
        origin_x = float(info.origin.position.x)
        origin_y = float(info.origin.position.y)

        data = np.array(self.map.data, dtype=np.int16).reshape((height, width))
        grid = np.full_like(data, fill_value=-1)
        grid[(data >= 0) & (data < 50)] = 0
        grid[data >= 50] = 1
        # (-1) permanece desconhecido

        self.map_grid = grid
        self.map_info = {
            'resolution': res,
            'width': width,
            'height': height,
            'origin_x': origin_x,
            'origin_y': origin_y,
        }

        # Seleciona células livres e amostra partículas
        livres = np.argwhere(self.map_grid == 0)
        if livres.size == 0:
            self.get_logger().warn('Mapa sem células livres para inicialização de partículas.')
            return

        choice_replace = livres.shape[0] < self.M
        idx = np.random.choice(livres.shape[0], self.M, replace=choice_replace)
        cells = livres[idx]  # [iy, ix]

        xs = origin_x + (cells[:, 1] + 0.5) * res
        ys = origin_y + (cells[:, 0] + 0.5) * res
        ths = np.random.uniform(-math.pi, math.pi, size=self.M)

        self.p[:, 0] = xs
        self.p[:, 1] = ys
        self.p[:, 2] = ths
        self.w[:] = 1.0 / self.M

        # Estimativa inicial (média ponderada)
        x_est = float(np.mean(xs))
        y_est = float(np.mean(ys))
        sin_sum = float(np.mean(np.sin(ths)))
        cos_sum = float(np.mean(np.cos(ths)))
        th_est = math.atan2(sin_sum, cos_sum)
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, th_est)
        self.mcl_pose.header.frame_id = 'map'
        self.mcl_pose.pose.pose.position.x = x_est
        self.mcl_pose.pose.pose.position.y = y_est
        self.mcl_pose.pose.pose.position.z = 0.0
        self.mcl_pose.pose.pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))

    # algorithm - atualização
    def mcl_algorithm(self):
        # Garante que há odometria, laser e mapa
        if self.odom is None or self.laser is None or self.map_grid is None:
            return

        self.get_logger().info('teste')
        # Etapa 1: Previsão 
        if self.last_odom is None:
            self.last_odom = self.odom
            return  # espera próximo ciclo para ter delta

        x0, y0, th0 = self.last_odom
        x1, y1, th1 = self.odom

        dx = x1 - x0
        dy = y1 - y0
        delta_trans = math.hypot(dx, dy)
        angle_wrap = lambda a: (a + math.pi) % (2.0 * math.pi) - math.pi
        delta_rot1 = angle_wrap(math.atan2(dy, dx) - th0) if delta_trans > 1e-6 else 0.0
        delta_rot2 = angle_wrap(th1 - th0 - delta_rot1)

        # Ruído nas componentes
        std_rot1 = math.sqrt(self.alpha1 * (delta_rot1**2) + self.alpha2 * (delta_trans**2)) + 1e-9
        std_trans = math.sqrt(self.alpha3 * (delta_trans**2) + self.alpha4 * (delta_rot1**2 + delta_rot2**2)) + 1e-9
        std_rot2 = math.sqrt(self.alpha1 * (delta_rot2**2) + self.alpha2 * (delta_trans**2)) + 1e-9

        # Amostrar para cada partícula
        rot1_s = delta_rot1 + np.random.normal(0.0, std_rot1, size=self.M)
        trans_s = max(0.0, delta_trans) + np.random.normal(0.0, std_trans, size=self.M)
        rot2_s = delta_rot2 + np.random.normal(0.0, std_rot2, size=self.M)

        # Atualiza partículas
        self.p[:,0] += trans_s * np.cos(self.p[:,2] + rot1_s)
        self.p[:,1] += trans_s * np.sin(self.p[:,2] + rot1_s)
        self.p[:,2] = (self.p[:,2] + rot1_s + rot2_s + math.pi) % (2.0 * math.pi) - math.pi

        self.last_odom = self.odom

        # Etapa 2: Correção (atualização por sensor)
        # Modelo de sensor: verossimilhança Gaussiana com distância esperada via raycast no mapa
        laser = self.laser
        max_range = float(laser.range_max) if self.max_range is None else float(self.max_range)
        if self.max_range is None:
            self.max_range = max_range
        n_ranges = len(laser.ranges)
        if n_ranges == 0:
            return
        if self.max_beams is None or self.max_beams <= 0 or self.max_beams > n_ranges:
            idxs = np.arange(n_ranges)
        else:
            idxs = np.linspace(0, n_ranges - 1, self.max_beams, dtype=int)

        angle_min = float(laser.angle_min)
        angle_inc = float(laser.angle_increment)
        z_sigma = max(self.z_sigma, 1e-3)
        two_sigma2 = 2.0 * (z_sigma ** 2)

        res = self.map_info['resolution']
        origin_x = self.map_info['origin_x']
        origin_y = self.map_info['origin_y']
        width = self.map_info['width']
        height = self.map_info['height']

        def world_to_map(wx, wy):
            ix = int((wx - origin_x) / res)
            iy = int((wy - origin_y) / res)
            return ix, iy

        def inside(ix, iy):
            return 0 <= ix < width and 0 <= iy < height

        def occupied(ix, iy):
            if not inside(ix, iy):
                return True
            v = self.map_grid[iy, ix]
            if v == -1:
                return True
            return v == 1

        log_w = np.zeros(self.M, dtype=float)

        for j in idxs:
            z = float(laser.ranges[j])
            if not np.isfinite(z) or z <= 0.0:
                continue
            beam_angle = angle_min + j * angle_inc
            ths = self.p[:, 2] + beam_angle
            z_exp = np.empty(self.M, dtype=float)
            # Raycast para cada partícula
            step = max(res * 0.5, 0.02)
            for i in range(self.M):
                cos_t = math.cos(ths[i])
                sin_t = math.sin(ths[i])
                dist = 0.0
                hit = max_range
                # Limita a busca um pouco acima da medida para eficiência
                local_max = min(max_range, z + 2.0 * z_sigma)
                while dist < local_max:
                    rx = self.p[i, 0] + dist * cos_t
                    ry = self.p[i, 1] + dist * sin_t
                    ix, iy = world_to_map(rx, ry)
                    if not inside(ix, iy):
                        hit = dist
                        break
                    if occupied(ix, iy):
                        hit = dist
                        break
                    dist += step
                z_exp[i] = hit
            dif = z - z_exp
            log_w += -(dif * dif) / two_sigma2

        # Penaliza partículas fora do mapa ou em célula inválida
        for i in range(self.M):
            ix, iy = world_to_map(self.p[i, 0], self.p[i, 1])
            if not inside(ix, iy) or occupied(ix, iy):
                log_w[i] -= 5.0

        # Normalização estável
        max_log = float(np.max(log_w))
        w = np.exp(log_w - max_log)
        s = float(np.sum(w))
        if s == 0.0 or not np.isfinite(s):
            w = np.ones(self.M) / self.M
        else:
            w /= s
        self.w = w

        # Etapa 3: Reamostragem (low-variance)
        M = self.M
        new_particles = np.zeros_like(self.p)
        r = np.random.uniform(0.0, 1.0 / M)
        c = self.w[0]
        i = 0
        invM = 1.0 / M
        for m in range(M):
            U = r + m * invM
            while U > c and i < M - 1:
                i += 1
                c += self.w[i]
            new_particles[m] = self.p[i]
        self.p = new_particles
        self.w[:] = invM

        # Etapa 4: Estimativa da posição (média ponderada)
        w = self.w
        w_sum = float(np.sum(w)) if w is not None else 0.0
        if w_sum <= 0:
            w = np.ones(M) / M
            w_sum = 1.0
        x_est = float(np.sum(self.p[:, 0] * w) / w_sum)
        y_est = float(np.sum(self.p[:, 1] * w) / w_sum)
        sin_sum = float(np.sum(np.sin(self.p[:, 2]) * w) / w_sum)
        cos_sum = float(np.sum(np.cos(self.p[:, 2]) * w) / w_sum)
        th_est = math.atan2(sin_sum, cos_sum)
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, th_est)
        
        self.mcl_pose.header.frame_id = 'map'
        self.mcl_pose.pose.pose.position.x = x_est
        self.mcl_pose.pose.pose.position.y = y_est
        self.mcl_pose.pose.pose.position.z = 0.0
        self.mcl_pose.pose.pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))


def main(args=None):
    rclpy.init(args=args) # Inicializando ROS
    node = MCL()          # Inicializando nó
    del node              # Finalizando nó
    rclpy.shutdown()      # Finalizando ROS

if __name__ == '__main__':
    main()
