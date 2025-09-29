SCRIPT="$0"
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

ros2 run gazebo_ros spawn_entity.py -file $SCRIPT_DIR/models/box.sdf -entity box_d -y 0 -x -1

ros2 run gazebo_ros spawn_entity.py -file $SCRIPT_DIR/models/box.sdf -entity box_e1 -y 1 -x 0
ros2 run gazebo_ros spawn_entity.py -file $SCRIPT_DIR/models/box.sdf -entity box_e2 -y 1 -x 1
ros2 run gazebo_ros spawn_entity.py -file $SCRIPT_DIR/models/box.sdf -entity box_e4 -y 1 -x 3  
ros2 run gazebo_ros spawn_entity.py -file $SCRIPT_DIR/models/box.sdf -entity box_e5 -y 1 -x 4

ros2 run gazebo_ros spawn_entity.py -file $SCRIPT_DIR/models/box.sdf -entity box_d1 -y -1 -x 0
ros2 run gazebo_ros spawn_entity.py -file $SCRIPT_DIR/models/box.sdf -entity box_d2 -y -1 -x 1
ros2 run gazebo_ros spawn_entity.py -file $SCRIPT_DIR/models/box.sdf -entity box_d4 -y -1 -x 3  
ros2 run gazebo_ros spawn_entity.py -file $SCRIPT_DIR/models/box.sdf -entity box_d5 -y -1 -x 4

ros2 run gazebo_ros spawn_entity.py -file $SCRIPT_DIR/models/box.sdf -entity box_t -y 0 -x 5

