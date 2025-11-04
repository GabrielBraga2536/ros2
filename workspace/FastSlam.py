#!/usr/bin/env python3
# coding: utf-8
"""
FastSLAM node (ROS2, rclpy)
Implementação baseada na última imagem do PDF fornecido pelo usuário.
Cada partícula tem seu mapa de landmarks (EKF por landmark).
Autores: adaptado para usuário
Referência: arquivo anexado. :contentReference[oaicite:1]{index=1}
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, Quaternion

import tf_transformations
import numpy as np
import math
import copy
from collections import namedtuple

# Estrutura de um landmark dentro de uma partícula
Landmark = namedtuple('Landmark', ['mu', 'sigma', 'observed'])

class FastSlamNode(Node):
    def __init__(self):
        super().__init__('fast_slam_node')
        self.get_logger().info('Inicializando FastSLAM node...')

        # QoS
        qos_map = QoSProfile(depth=10)
        qos_map.reliability = QoSReliabilityPolicy.RELIABLE
        qos_map.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        qos_scan = QoSProfile(depth=10)
        qos_scan.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # Subscriptions
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_map)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_scan)

        # Publishers
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/fastslam_pose', 10)
        self.pcloud_pub = self.create_publisher(PoseArray, '/particlecloud', 10)
        self.map_pub = self.create_publisher(PoseArray, '/fastslam_map', 10)  # landmarks do melhor particle

        # Parâmetros
        self.declare_parameter('particles', 200)
        self.M = int(self.get_parameter('particles').value)

        self.declare_parameter('seed', None)
        seed = self.get_parameter('seed').value
        self.rng = np.random.default_rng(seed if seed is not None else None)

        # Alphas para o modelo de odometria (Thrun/Burgard/Fox)
        self.declare_parameter('alpha1', 0.1)  # rot noise proportional to rot
        self.declare_parameter('alpha2', 0.1)  # rot noise proportional to trans
        self.declare_parameter('alpha3', 0.2)  # trans noise proportional to trans
        self.declare_parameter('alpha4', 0.1)  # trans noise proportional to rot
        self.alpha1 = float(self.get_parameter('alpha1').value)
        self.alpha2 = float(self.get_parameter('alpha2').value)
        self.alpha3 = float(self.get_parameter('alpha3').value)
        self.alpha4 = float(self.get_parameter('alpha4').value)

        # Ruído do sensor (range, bearing)
        self.declare_parameter('r_sigma', 0.2)
        self.declare_parameter('b_sigma', 0.05)
        self.r_sigma = float(self.get_parameter('r_sigma').value)
        self.b_sigma = float(self.get_parameter('b_sigma').value)
        self.R = np.diag([self.r_sigma**2, self.b_sigma**2])

        # Associação
        self.declare_parameter('assoc_threshold', 7.8)  # Mahalanobis threshold
        self.assoc_thresh = float(self.get_parameter('assoc_threshold').value)

        # Scan handling
        self.declare_parameter('max_beams', 60)
        self.max_beams = int(self.get_parameter('max_beams').value)

        # Estado interno
        self.particles = []  # cada partícula é dict {'pose': np.array([x,y,th]), 'landmarks': [Landmark,...], 'weight': float}
        self.weights = np.ones(self.M) / float(self.M)
        self.initialized = False

        # Odometry
        self.last_odom = None  # nav_msgs/Odometry -> (x,y,th)
        self.odom = None

        # Laser
        self.scan = None
        self.max_range = None

        # Timer
        self.dT = 0.1
        self.timer = self.create_timer(self.dT, self.timer_callback)

        # Inicializa partículas com pose (0,0,0) por padrão; usuário pode melhorar amostrando do mapa
        for i in range(self.M):
            p = {
                'pose': np.array([0.0, 0.0, 0.0], dtype=float),
                'landmarks': [],
                'weight': 1.0 / self.M
            }
            self.particles.append(p)
        self.weights[:] = 1.0 / self.M
        self.initialized = True
        self.get_logger().info(f'FastSLAM: {self.M} partículas inicializadas (poses zeradas).')

    # ---------------- callbacks ----------------
    def odom_callback(self, msg: Odometry):
        # extrai pose da odometria
        q = msg.pose.pose.orientation
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.odom = np.array([x, y, yaw], dtype=float)
        if self.last_odom is None:
            self.last_odom = self.odom.copy()

    def scan_callback(self, msg: LaserScan):
        self.scan = msg
        if self.max_range is None:
            self.max_range = float(msg.range_max)

    # ---------------- main timer ----------------
    def timer_callback(self):
        if not self.initialized or self.odom is None or self.scan is None:
            return

        # Calcula delta de odometria entre chamadas (no frame do mapa)
        odom_cur = self.odom
        odom_prev = self.last_odom if self.last_odom is not None else odom_cur.copy()
        delta = self.compute_odom_delta(odom_prev, odom_cur)
        self.last_odom = odom_cur.copy()

        # PREDICT para cada partícula (usando modelo de odometria com ruido)
        for p in self.particles:
            self.motion_update_particle(p, delta)

        # UPDATE: extrai observações do laser (samples) e atualiza landmarks/weights por partícula
        observations = self.extract_observations(self.scan)
        if len(observations) > 0:
            for i, p in enumerate(self.particles):
                w = self.update_particle_with_observations(p, observations)
                p['weight'] = w
                self.weights[i] = w

        # Normaliza pesos
        wsum = float(np.sum(self.weights))
        if wsum <= 0 or not np.isfinite(wsum):
            self.get_logger().warning('Peso inválido detectado; resetando pesos uniformes.')
            self.weights[:] = 1.0 / self.M
            for i, p in enumerate(self.particles):
                p['weight'] = self.weights[i]
        else:
            self.weights /= wsum
            for i, p in enumerate(self.particles):
                p['weight'] = self.weights[i]

        # Reamostragem sistemática se necessário
        nef = 1.0 / np.sum(self.weights ** 2)
        if nef < 0.5 * self.M:
            self.resample_systematic()
            self.get_logger().info(f'Reamostragem executada (Neff={nef:.1f}).')

        # Estimativa da pose (média ponderada)
        est_pose = self.estimate_pose()

        # Publicações
        self.publish_pose(est_pose)
        self.publish_particles()
        self.publish_best_map()

    # ---------------- odom delta ----------------
    def compute_odom_delta(self, prev, cur):
        # prev, cur são [x,y,th] em mapa.
        dx = cur[0] - prev[0]
        dy = cur[1] - prev[1]
        # transformar deslocamento em frame do prev
        dtrans = math.hypot(dx, dy)
        dtheta = (cur[2] - prev[2] + math.pi) % (2*math.pi) - math.pi

        # compute rot1 = atan2(dy,dx) - prev_th
        rot1 = math.atan2(dy, dx) - prev[2]
        rot1 = (rot1 + math.pi) % (2*math.pi) - math.pi
        rot2 = dtheta - rot1
        rot2 = (rot2 + math.pi) % (2*math.pi) - math.pi

        return np.array([rot1, dtrans, rot2], dtype=float)  # (rot1, trans, rot2)

    # ---------------- motion model (sample) ----------------
    def motion_update_particle(self, p, odom_delta):
        # odom_delta: [rot1, trans, rot2]
        rot1, trans, rot2 = odom_delta
        # adicionar ruido aos componentes (Thrun)
        std_rot1 = math.sqrt(self.alpha1 * rot1**2 + self.alpha2 * trans**2)
        std_trans = math.sqrt(self.alpha3 * trans**2 + self.alpha4 * (rot1**2 + rot2**2))
        std_rot2 = math.sqrt(self.alpha1 * rot2**2 + self.alpha2 * trans**2)

        # sample ruídos
        r1_hat = rot1 + self.rng.normal(0.0, std_rot1)
        t_hat = trans + self.rng.normal(0.0, std_trans)
        r2_hat = rot2 + self.rng.normal(0.0, std_rot2)

        # atualizar pose da partícula
        x, y, th = p['pose']
        x_new = x + t_hat * math.cos(th + r1_hat)
        y_new = y + t_hat * math.sin(th + r1_hat)
        th_new = th + r1_hat + r2_hat
        th_new = (th_new + math.pi) % (2*math.pi) - math.pi
        p['pose'] = np.array([x_new, y_new, th_new], dtype=float)

    # ---------------- extrair observações do scan ----------------
    def extract_observations(self, scan_msg):
        ranges = np.array(scan_msg.ranges, dtype=float)
        n = len(ranges)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, n, dtype=float)
        step = max(1, n // self.max_beams)
        obs = []  # cada obs = (r, bearing)
        for a, r in zip(angles[::step], ranges[::step]):
            if not np.isfinite(r):
                continue
            if r <= 0.01 or r > self.max_range:
                continue
            obs.append((float(r), float(a)))
        return obs

    # ---------------- update particle com observações (EKF por landmark) ----------------
    def update_particle_with_observations(self, p, observations):
        x, y, th = p['pose']
        landmarks = p['landmarks']
        prev_weight = p.get('weight', 1.0 / self.M)
        log_like = 0.0

        for (r_meas, b_meas) in observations:
            # posição estimada do landmark no mapa (observação convertida)
            lx = x + r_meas * math.cos(th + b_meas)
            ly = y + r_meas * math.sin(th + b_meas)
            z = np.array([r_meas, b_meas], dtype=float)

            # data association: procurar melhor landmark pela menor Mahalanobis
            best_i = None
            best_maha = None
            best_S = None
            best_zhat = None

            for i, lm in enumerate(landmarks):
                mu = lm.mu
                sigma = lm.sigma
                dx = mu[0] - x
                dy = mu[1] - y
                q = dx*dx + dy*dy
                if q <= 1e-8:
                    continue
                sqrt_q = math.sqrt(q)
                # previsão de medição z_hat = [range, bearing]
                z_hat = np.array([sqrt_q, math.atan2(dy, dx) - th], dtype=float)
                z_hat[1] = (z_hat[1] + math.pi) % (2*math.pi) - math.pi

                # Jacobiana H (w.r.t. landmark position mu)
                H = np.array([[dx / sqrt_q, dy / sqrt_q],
                              [-dy / q,       dx / q      ]], dtype=float)

                S = H @ sigma @ H.T + self.R
                innov = z - z_hat
                innov[1] = (innov[1] + math.pi) % (2*math.pi) - math.pi

                # Mahalanobis
                try:
                    invS = np.linalg.inv(S)
                    maha = float(innov.T @ invS @ innov)
                except np.linalg.LinAlgError:
                    maha = np.inf

                if best_i is None or maha < best_maha:
                    best_i = i
                    best_maha = maha
                    best_S = S
                    best_zhat = z_hat

            # Decisão de associação
            if best_i is not None and best_maha is not None and best_maha < self.assoc_thresh:
                # Atualiza landmark via EKF
                lm = landmarks[best_i]
                mu = lm.mu
                sigma = lm.sigma

                dx = mu[0] - x
                dy = mu[1] - y
                q = dx*dx + dy*dy
                if q <= 1e-8:
                    continue
                sqrt_q = math.sqrt(q)
                z_hat = np.array([sqrt_q, math.atan2(dy, dx) - th], dtype=float)
                z_hat[1] = (z_hat[1] + math.pi) % (2*math.pi) - math.pi

                H = np.array([[dx / sqrt_q, dy / sqrt_q],
                              [-dy / q,       dx / q      ]], dtype=float)
                S = H @ sigma @ H.T + self.R
                try:
                    K = sigma @ H.T @ np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    # se S singular, penalize and skip update
                    log_like += -50.0
                    continue

                innov = z - z_hat
                innov[1] = (innov[1] + math.pi) % (2*math.pi) - math.pi

                mu_new = mu + K @ innov
                sigma_new = (np.eye(2) - K @ H) @ sigma
                landmarks[best_i] = Landmark(mu=mu_new, sigma=sigma_new, observed=lm.observed + 1)

                # Likelihood gaussian
                try:
                    detS = np.linalg.det(S)
                    denom = 2 * math.pi * math.sqrt(detS) if detS > 0 else 1e-6
                    exponent = -0.5 * (innov.T @ np.linalg.inv(S) @ innov)
                    lik = math.exp(exponent) / denom
                    log_like += math.log(lik + 1e-12)
                except np.linalg.LinAlgError:
                    log_like += -50.0

            else:
                # Inicializa novo landmark a partir da medição
                mu = np.array([lx, ly], dtype=float)
                dx = mu[0] - x
                dy = mu[1] - y
                q = dx*dx + dy*dy
                sqrt_q = math.sqrt(q) if q > 0 else 1e-6

                H = np.array([[dx / sqrt_q, dy / sqrt_q],
                              [-dy / q if q>0 else 0.0, dx / q if q>0 else 0.0]], dtype=float)
                # tentativa de projetar R para espaço do landmark: sigma = inv(H) * R * inv(H).T
                try:
                    invH = np.linalg.inv(H)
                    sigma = invH @ self.R @ invH.T
                except np.linalg.LinAlgError:
                    sigma = np.eye(2) * 1.0

                landmarks.append(Landmark(mu=mu, sigma=sigma, observed=1))
                # pequena contribuição de verossimilhança inicial
                log_like += math.log(1e-3)

        new_weight = prev_weight * math.exp(log_like)
        if not np.isfinite(new_weight) or new_weight <= 0:
            new_weight = 1e-12
        return new_weight

    # ---------------- resample (systematic) ----------------
    def resample_systematic(self):
        M = self.M
        weights = self.weights
        positions = (self.rng.random() + np.arange(M)) / M
        cumulative = np.cumsum(weights)
        new_particles = []
        i = 0
        for pos in positions:
            while pos > cumulative[i]:
                i += 1
            chosen = copy.deepcopy(self.particles[i])
            chosen['weight'] = 1.0 / M
            new_particles.append(chosen)
        self.particles = new_particles
        self.weights[:] = 1.0 / M

    # ---------------- estimate pose (weighted mean) ----------------
    def estimate_pose(self):
        xs = np.array([p['pose'][0] for p in self.particles], dtype=float)
        ys = np.array([p['pose'][1] for p in self.particles], dtype=float)
        thetas = np.array([p['pose'][2] for p in self.particles], dtype=float)
        w = self.weights
        mx = float(np.sum(w * xs))
        my = float(np.sum(w * ys))
        cx = float(np.sum(w * np.cos(thetas)))
        sx = float(np.sum(w * np.sin(thetas)))
        mth = math.atan2(sx, cx)
        return np.array([mx, my, mth], dtype=float)

    # ---------------- publishers ----------------
    def publish_pose(self, est_pose):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(est_pose[0])
        msg.pose.pose.position.y = float(est_pose[1])
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, float(est_pose[2]))
        msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        # covariância simples (substitua por cálculo a partir da distribuição de partículas se desejar)
        cov = np.zeros((6,6), dtype=float)
        cov[0,0] = 0.1
        cov[1,1] = 0.1
        cov[5,5] = 0.5
        msg.pose.covariance = [float(x) for x in cov.flatten()]
        self.pose_pub.publish(msg)

    def publish_particles(self):
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = 'map'
        for p in self.particles:
            pose = Pose()
            pose.position.x = float(p['pose'][0])
            pose.position.y = float(p['pose'][1])
            q = tf_transformations.quaternion_from_euler(0.0, 0.0, float(p['pose'][2]))
            pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            pa.poses.append(pose)
        self.pcloud_pub.publish(pa)

    def publish_best_map(self):
        # publica landmarks do particle com maior peso
        best_idx = int(np.argmax(self.weights))
        best = self.particles[best_idx]
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = 'map'
        for lm in best['landmarks']:
            pose = Pose()
            pose.position.x = float(lm.mu[0])
            pose.position.y = float(lm.mu[1])
            pose.position.z = 0.0
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            pa.poses.append(pose)
        self.map_pub.publish(pa)


def main(args=None):
    rclpy.init(args=args)
    node = FastSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down FastSLAM node...')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
