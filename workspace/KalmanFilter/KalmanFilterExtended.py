import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion

from tf_transformations import quaternion_from_euler

import numpy as np  

class EKF(Node):

  # Inicializando nó
  def __init__(self):
    super().__init__('EKF')
    self.get_logger().info('Inicializando o nó!')

    self.create_subscription(Odometry, '/odom', self.subscriber_callback, 10)
    self.publisher_pose = self.create_publisher(Pose, '/pose', 10)

    ########## inicializando variáveis ##########
    self.dT = 1.0  # intervalo de tempo (s)
    
    # Estado [x, y, theta]^T
    self.x = np.array([[0.0], [0.0], [0.0]])  
    
    # Controle [velocidade linear, velocidade angular]^T
    self.u = np.array([[0.0], [0.0]])    
    
    # Covariância do estado (3x3)
    self.P = np.eye(3) * 0.001  
    
    # Ruído do processo (modelo)
    self.Q = np.eye(3) * 0.001  
    
    # Medição [x_meas, y_meas]^T
    self.z = np.array([[0.0], [0.0]])
    
    # Ruído da medição (sensor)
    self.R = np.diag([0.01, 0.01])
    #############################################

    self.timer = self.create_timer(self.dT, self.timer_callback)
    rclpy.spin(self)

  # Finalizando nó
  def _del_(self):
    self.get_logger().info('Finalizando o nó!')
    self.destroy_node()

  # Odom callback
  def subscriber_callback(self, msg):
    # Atualiza a medição de posição (x, y)
    self.z = np.array([[msg.pose.pose.position.x], [msg.pose.pose.position.y]])
    
    # Atualiza a matriz de covariância da medição (usando valores do covariance do Odometry)
    self.R = np.diag([msg.pose.covariance[0], msg.pose.covariance[7]])
    
    # Atualiza o controle: velocidade linear e angular
    self.u = np.array([[msg.twist.twist.linear.x], [msg.twist.twist.angular.z]])

  # Executando o filtro no timer
  def timer_callback(self):        
    self.x, self.P = self.ekf_algorithm(self.x, self.P, self.u, self.z, self.Q, self.R)

    msg = Pose()
    msg.position = Point(x=float(self.x[0].item()), y=float(self.x[1].item()), z=0.0)

    qx, qy, qz, qw = quaternion_from_euler(0, 0, self.x[2].item())
    msg.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        
    self.publisher_pose.publish(msg)

  # EKF: predição e atualização
  def ekf_algorithm(self, x, P, u, z, Q, R):
    # Extrai valores do estado e controle
    x_pos, y_pos, theta = x.flatten()
    v, w = u.flatten()
    dt = self.dT
    
    # Modelo de movimento (predição)
    if abs(w) > 1e-5:
      x_pred = x_pos + (v / w) * (np.sin(theta + w*dt) - np.sin(theta))
      y_pred = y_pos + (v / w) * (-np.cos(theta + w*dt) + np.cos(theta))
      theta_pred = theta + w * dt
    else:
      x_pred = x_pos + v * dt * np.cos(theta)
      y_pred = y_pos + v * dt * np.sin(theta)
      theta_pred = theta
    
    x_pred = np.array([[x_pred], [y_pred], [theta_pred]])
    h_pred = np.array([[x_pred], [y_pred]])  
    
    # Jacobiana F_t (derivada do modelo em relação ao estado)
    F_t = np.array([
      [1, 0, -v*dt*np.sin(theta)],
      [0, 1,  v*dt*np.cos(theta)],
      [0, 0, 1]
    ])
    
    # Jacobiana da observação H_t (medimos só x e y)
    H_t = np.array([
      [1, 0, 0],
      [0, 1, 0]
    ])
    
    # Covariância da predição
    P_pred = F_t @ P @ F_t.T + Q
    
    # Inovação (resíduo)
    y_tilde = z - (H_t @ x_pred)
    
    # Covariância do resíduo
    S = H_t @ P_pred @ H_t.T + R
    
    # Ganho de Kalman
    K = P_pred @ H_t.T @ np.linalg.inv(S)
    
    # Atualização do estado
    x_new = x_pred + K @ y_tilde
    
    # Atualização da covariância
    P_new = (np.eye(3) - K @ H_t) @ P_pred
    
    self.get_logger().info(f"[EKF] X: {np.array2string(x_new.flatten(), separator=', ')}")
    self.get_logger().info(f"[EKF] Z: {np.array2string(self.z.flatten(), separator=', ')}")
    
    return x_new, P_new

def main(args=None):
  rclpy.init(args=args)  # Inicializando ROS
  node = EKF()           # Inicializando nó
  del node               # Finalizando nó
  rclpy.shutdown()       # Finalizando ROS

if __name__ == '__main__':
  main()