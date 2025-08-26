import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np


class MarkovLocalizationNode(Node):
    def __init__(self, num_states=5):
        super().__init__('markov_localization_node')
        self.get_logger().info('Inicializando nó de Localização de Markov!')

        # Publisher do belief
        self.publisher = self.create_publisher(Float64MultiArray, "belief", 100)

        # Número de estados (posições possíveis)
        self.num_states = num_states

        # Inicialização uniforme: robô não sabe onde está
        self.bel = np.ones(num_states) / num_states

        # Timer para rodar o filtro periodicamente
        self.create_timer(1.0, self.timer_callback)

    def transition_model(self, x, x_prev, u):
        """Modelo de transição P(x | u, x_prev)"""
        if u == "right":
            return 1.0 if x == (x_prev + 1) % self.num_states else 0.0
        elif u == "left":
            return 1.0 if x == (x_prev - 1) % self.num_states else 0.0
        else:
            return 1.0 if x == x_prev else 0.0

    def sensor_model(self, z, x):
        """Modelo de observação P(z | x)"""
        return 0.8 if z == x else 0.2 / (self.num_states - 1)

    def markov_update(self, u, z):
        # 1. Predição
        bel_pred = np.zeros(self.num_states)
        for x in range(self.num_states):
            for x_prev in range(self.num_states):
                bel_pred[x] += self.transition_model(x, x_prev, u) * self.bel[x_prev]

        # 2. Correção
        bel_new = np.zeros(self.num_states)
        for x in range(self.num_states):
            bel_new[x] = self.sensor_model(z, x) * bel_pred[x]

        # 3. Normalização
        bel_new /= np.sum(bel_new)
        self.bel = bel_new

    def timer_callback(self):
        # Exemplo: cada passo o robô tenta ir para a direita e o sensor "detecta" a posição 2
        u = "right"
        z = 2

        self.markov_update(u, z)

        # Publicar a crença
        msg = Float64MultiArray()
        msg.data = self.bel.tolist()
        self.publisher.publish(msg)

        self.get_logger().info(f"Crença atual (Markov): {self.bel}")


def main(args=None):
    rclpy.init(args=args)
    node = MarkovLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()