#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Vector3
import numpy as np

class MarkovLocalization(Node):

    # Inicializando nó
    def __init__(self):
        super().__init__('markov_localization')
        self.get_logger().info('Inicializando o nó!')

        # Inicialização da crença
        self.bel = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        # Inicialzação da porta
        self.door = [False, False, True, False, False]

        self.p_ok = 0.8             # Probabilidade de acerto na ação
        self.p_err = 1 - self.p_ok  # Probabilidade de erro na ação

        self.p_is_door = 0.9        # Confiança de ser uma porta
        self.p_no_door = 0.1        # Confiança se não for uma porta

        self.robot_front_laser = None
        self.robot_side_laser = None
        self.subscription = self.create_subscription(LaserScan, '/scan', self.subscriber_callback, 10)

        self.publisher_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.timer_period = 5.0
        self.timer = self.create_timer(self.timer_period,self.timer_callback)
        rclpy.spin(self)

    # Finalizando nó
    def __del__(self):
        self.get_logger().info('Finalizando o nó!')
        self.destroy_node()

    def subscriber_callback(self, msg):
        self.robot_front_laser = msg.ranges[0]
        self.robot_side_laser = msg.ranges[90]

    def timer_callback(self):
        msg_stop = Twist(linear=Vector3(x=0.0,y=0.0,z=0.0), angular=Vector3(x=0.0,y=0.0,z=0.0))
        msg_forw = Twist(linear=Vector3(x=0.2,y=0.0,z=0.0), angular=Vector3(x=0.0,y=0.0,z=0.0))

        self.publisher_vel.publish(msg_stop)

        self.get_logger().info(f'Valor do laser lateral do robô: {self.robot_side_laser}')

        self.solution()

        if(self.robot_front_laser > 1.0):
            self.publisher_vel.publish(msg_forw)

    def solution(self):
        new_bel = self.bel.copy()

        for i in range(len(new_bel)):
            if(i == 0):
                ok_anterior = 0.0
            else:
                ok_anterior = self.bel[i-1] * self.p_ok

            if(i==(len(new_bel)-1)):
                err_atual = self.bel[i]
            else:
                err_atual = self.bel[i] * self.p_err
            
            new_bel[i] = ok_anterior + err_atual
        
        self.get_logger().info(f'Nova crença com movimento: {new_bel}')

        for i in range(len(new_bel)):
            if(self.door[i]):
                new_bel[i] = new_bel[i] * self.p_is_door
            else:
                new_bel[i] = new_bel[i] * self.p_no_door

        self.get_logger().info(f'Nova crença com o sensor: {new_bel}')

        soma = sum(new_bel)
        for i in range(len(new_bel)):
            new_bel[i] = new_bel[i] / soma
        
        self.get_logger().info(f'Nova crença normalizada: {new_bel}')

        self.bel = new_bel 



def main(args=None):
    rclpy.init(args=args) # Inicializando ROS
    node = MarkovLocalization() # Inicializando nó
    del node              # Finalizando nó
    rclpy.shutdown()      # Finalizando ROS

if __name__ == '__main__':
    main()
