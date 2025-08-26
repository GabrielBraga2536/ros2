import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class MyNode(Node):
  
  def __init__(self):
    super().__init__('no_com_classe')
    self.get_logger().info('Inicializando o nó!')
    self.publisher = self.create_publisher(Twist, "cmd_vel", 100)
  
  def __del__(self):
    self.get_logger().info('Finalizando o nó!')
  
  def execute(self):
    self.get_logger().info('Executando o nó!')
    self.create_timer(1, self.timer_callback)
    rclpy.spin(self)
  
  def timer_callback(self):
    self.talker_publisher()
  
  def talker_publisher(self):
    vel = Twist()
    vel.linear.x = 0.1
    self.publisher.publish(vel)

  
def main(args=None):
  rclpy.init(args=args)
  my_node = MyNode()
  my_node.execute()

if __name__ == '__main__':
  main()
