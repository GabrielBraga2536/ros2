import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MyNode(Node):
  
  def __init__(self):
    super().__init__('no_com_classe')
    self.get_logger().info('Inicializando o nó!')
    self.publisher_1 = self.create_publisher(String, "topico", 10)
    self.publisher_2 = self.create_publisher(String, "topico2", 10)

  def __del__(self):
    self.get_logger().info('Finalizando o nó!')
  
  def execute(self):
    self.get_logger().info('Executando o nó!')
    self.create_timer(1, self.timer_callback)
    rclpy.spin(self)
  
  def timer_callback(self):
    self.talker_publisher_1("hello world")
    self.talker_publisher_2("world hello")
  
  def talker_publisher_1(self, data):
    msg = String()
    msg.data = data
    self.publisher_1.publish(msg)
  
  def talker_publisher_2(self, data):
    msg = String()
    msg.data = data
    self.publisher_2.publish(msg)

def main(args=None):
  rclpy.init(args=args)
  my_node = MyNode()
  my_node.execute()

if __name__ == '__main__':
  main()
