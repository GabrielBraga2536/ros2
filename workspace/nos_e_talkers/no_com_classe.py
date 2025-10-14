import rclpy
from rclpy.node import Node

class MyNode(Node):
  
  def __init__(self):
    super().__init__('no_com_classe')
    self.get_logger().info('Inicializando o nó!')
  
  def __del__(self):
    self.get_logger().info('Finalizando o nó!')
  
  def execute(self):
    self.get_logger().info('Executando o nó!')
    
    self.get_logger().debug ('Exemplo de mensagem de debug.')
    self.get_logger().info  ('Exemplo de mensagem de informação.')
    self.get_logger().warn  ('Exemplo de mensagem de aviso.')
    self.get_logger().error ('Exemplo de mensagem de erro comum.')
    self.get_logger().fatal ('Exemplo de mensagem de erro fatal.')
    
    rclpy.spin(self)

def main(args=None):
  rclpy.init(args=args)
  my_node = MyNode()
  my_node.execute()

if __name__ == '__main__':
  main()
