#!/bin/bash
SCRIPT="$0"
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "name: k404" > /dev/null

ros2 run gazebo_ros spawn_entity.py -file $SCRIPT_DIR/models/k404.sdf -entity k404 -x 4.5 -y -0.5 -Y 1.5708

