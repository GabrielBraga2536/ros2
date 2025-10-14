import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class MyNode(Node):
  
  def __init__(self):
    super().__init__('no_com_classe')
    self.get_logger().info('Inicializando o nó!')
    
    qos_profile = QoSProfile(depth=10, reliability = QoSReliabilityPolicy.BEST_EFFORT)
    self.subscription = self.create_subscription(String, 'topico', self.listener_callback, qos_profile )
  
  def __del__(self):
    self.get_logger().info('Finalizando o nó!')
  
  def execute(self):
    self.get_logger().info('Executando o nó!')
    rclpy.spin(self)
  
  def listener_callback(self, msg):
    self.get_logger().info('Mensagem publicada: "%s"' % msg.data)

def main(args=None):
  rclpy.init(args=args)
  my_node = MyNode()
  my_node.execute()

if __name__ == '__main__':
  main()
