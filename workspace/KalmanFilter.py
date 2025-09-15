import rclpy
from rclpy.node import Node
import numpy as np

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64 

import numpy as np  

class KalmanFilter(Node):

    # Inicializando nó
    def __init__(self):
        super().__init__('kalman_filter')
        self.get_logger().info('Inicializando o nó!')

        self.px = 0.0
        self.cov_px = 0.01
        self.vx = 0.0
        self.con_vx = 0.01
        self.create_subscription(Odometry, '/odom', self.subscriber_callback, 10)

        self.publisher_x = self.create_publisher(Float64, '/x', 10)

        self.z = 0.0

        self.dT = 0.5
        self.timer = self.create_timer(self.dT,self.timer_callback)

        self.start()

        rclpy.spin(self)

    # Finalizando nó
    def __del__(self):
        self.get_logger().info('Finalizando o nó!')
        self.destroy_node()

    def subscriber_callback(self, msg):
        self.px = msg.pose.pose.position.x
        self.cov_px = msg.pose.covariance[0]
        self.vx = msg.twist.twist.linear.x
        self.cov_vx = msg.twist.covariance[0]

    # Executando nó
    def timer_callback(self):
        # Atualiza z com a posição atual antes de chamar o filtro
        self.z = np.array([[self.px]])
        [self.x, self.P] = self.KF(self.x, self.P, self.u, self.z)
        # self.get_logger().info(f'Posição estimada: {self.x[0].item()} m')
        # self.get_logger().info(f'Velocidade estimada: {self.x[1].item()} m/s')
        msg = Float64()
        msg.data = self.x[0].item()
        self.publisher_x.publish(msg)
        

    # inicialização
    def start(self):
        # inicializar matrizes

        # Estado [x; vx]
        self.x = np.array([ [0.0],
                            [0.0]])

        # Covariância inicial
        # np.eye inicializa uma matriz identidade
        self.P = np.eye(2) * 0.001

        # Matrizes do modelo
        self.A = np.array([ [1.0, self.dT],
                            [0.0, 1.0]])  # transição de estados

        self.B = np.array([ [0.5 * (self.dT**2)],
                            [self.dT]])   # controle (aceleração)

        # Medimos posição e velocidade do /odom
        self.H = np.array([[1.0, 0.0]])

        # Ruídos (ajuste conforme seu sistema)
        self.Q = np.eye(2) * 0.001  # ruído de processo
        # self.R = np.eye(2) * 0.01   # ruído de medição
        self.R = np.array([[0.01]])

        # Vetores de controle e medição
        self.u = np.array([[0.0]])         # aceleração
        self.z = np.array([[self.px]])     # apenas posição
        
    # atualização
    def KF(self,x,P,u,z):
        # impementar algoritmo

        # Previsão do estado: x̄ = A·x + B·u
        x_pred = self.A @ x + self.B @ u

        # Previsão da incerteza: P̄ = A·P·A^T + Q
        P_pred = self.A @ P @ self.A.T + self.Q

        # --- 2. CORREÇÃO ---
        # Ganho de Kalman: K = P̄·H^T·(H·P̄·H^T + R)^(-1)
        S = self.H @ P_pred @ self.H.T + self.R    # matriz da inovação
        K = P_pred @ self.H.T @ np.linalg.inv(S)   # ganho de Kalman

        # Atualização do estado: x = x̄ + K·(z - H·x̄)
        y = z - (self.H @ x_pred)   # inovação (diferença da medição)
        x = x_pred + K @ y

        # Atualização da incerteza: P = (I - K·H)·P̄
        I = np.eye(P.shape[0])      # matriz identidade
        P = (I - K @ self.H) @ P_pred

        self.get_logger().info(f' X: {x}')
        self.get_logger().info(f' Z: {self.z}')
        return [x,P]

def main(args=None):
    rclpy.init(args=args) # Inicializando ROS
    node = KalmanFilter() # Inicializando nó
    del node              # Finalizando nó
    rclpy.shutdown()      # Finalizando ROS

if __name__ == '__main__':
    main()
