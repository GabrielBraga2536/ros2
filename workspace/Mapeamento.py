import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

class Mapear(Node):
  
  def __init__(self):
    super().__init__("mapeamento")
    
    qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
    
    self.laser = None
    self.create_subscription(LaserScan, '/scan', self.listener_callback_laser, qos_profile)

    self.pose = None
    self.create_subscription(Odometry, '/odom', self.listener_callback_odom, qos_profile)
    
    self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
  
  def listener_callback_laser(self, msg):
    self.laser = msg.ranges
  
  def listener_callback_odom(self, msg):
    self.pose = msg.pose.pose
  
  def wait(self, max_seconds):
    start = time.time()
    count = 0
    while count < max_seconds:
      count = time.time() - start
      rclpy.spin_once(self)
  
  def navigantion_start(self):
    self.ir_para_frente = Twist(
      linear=Vector3(x=0.15, y=0.0, z=0.0),
      angular=Vector3(x=0.0, y=0.0, z=0.0))
    
    self.ir_para_tras = Twist(
      linear=Vector3(x=-0.15, y=0.0, z=0.0),
      angular=Vector3(x=0.0, y=0.0, z=0.0))
    
    self.girar_direita = Twist(
      linear=Vector3(x=-0.0, y=0.0, z=0.0),
      angular=Vector3(x=1.0, y=0.0, z=-0.22))
    
    self.girar_esquerda = Twist(
      linear=Vector3(x=0.0, y=0.0, z=0.0),
      angular=Vector3(x=0.0, y=0.0, z=0.22))
    
    self.parar = Twist(
      linear=Vector3(x=0.0, y=0.0, z=0.0),
      angular=Vector3(x=0.0, y=0.0, z=0.0))
    
    self.curva_direita = Twist(
      linear=Vector3(x=0.1, y=0.0, z=0.0),
      angular=Vector3(x=0.0, y=0.0, z=-0.15))
    
    self.curva_esquerda = Twist(
      linear=Vector3(x=0.1, y=0.0, z=0.0),
      angular=Vector3(x=0.0, y=0.0, z=0.15))

  def navigation_uptade(self):
    
    distancia_direita = min((self.laser[0:80]))
    distancia_frente = min((self.laser[80:100]))
    distancia_esquerda = min((self.laser[100:180]))
    
    if distancia_frente > 1.5:
      self.pub_cmd_vel.publish(self.ir_para_frente)
    elif distancia_frente > 0.75:
      if distancia_direita < distancia_esquerda:
        self.pub_cmd_vel.publish(self.curva_esquerda)
      else:
        self.pub_cmd_vel.publish(self.curva_direita)
    else:
      if distancia_direita < distancia_esquerda:
        self.pub_cmd_vel.publish(self.girar_esquerda)
      else:
        self.pub_cmd_vel.publish(self.girar_direita)
  
  def run(self):
    try:
      rclpy.spin_once(self)
      self.navigantion_start()
      while rclpy.ok:
        rclpy.spin_once(self)
        self.navigation_uptade()
    except KeyboardInterrupt:
      pass

def main(args=None):
  rclpy.init(args=args)
  node = Mapear()
  try:
    node.run()
    node.destroy_node()
    rclpy.shutdown()
  except KeyboardInterrupt:
    pass

if __name__ == '__main__':
  main()
