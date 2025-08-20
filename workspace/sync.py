import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_msgs.msg import String

class MyNode(Node):
  
  def __init__(self):
    super().__init__('no_com_classe')
    self.get_logger().info('Inicializando o nó!')
    
    self.subscriber_1 = Subscriber(self, String, 'topico')
    self.subscriber_2 = Subscriber(self, String, 'topico2')
    
    self.sync = ApproximateTimeSynchronizer([self.subscriber_1, self.subscriber_2], 10, 0.2, allow_headerless=True)
    self.sync.registerCallback(self.listener_callback)
  
  def __del__(self):
    self.get_logger().info('Finalizando o nó!')
  
  def execute(self):
    self.get_logger().info('Executando o nó!')
    
    rclpy.spin(self)
  
  def listener_callback(self, msg_1, msg_2):
    self.get_logger().info('Mensagem 1: "%s"' % msg_1.data)
    self.get_logger().info('Mensagem 2: "%s"' % msg_2.data)

def main(args=None):
  rclpy.init(args=args)
  my_node = MyNode()
  my_node.execute()

if __name__ == '__main__':
  main()