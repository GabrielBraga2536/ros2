#!/bin/bash
SCRIPT="$0"
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "name: room" > /dev/null
ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "name: box" > /dev/null

ros2 run gazebo_ros spawn_entity.py -file $SCRIPT_DIR/models/room.sdf -entity room -x 0 -y 0
ros2 run gazebo_ros spawn_entity.py -file $SCRIPT_DIR/models/box.sdf  -entity box -x -6 -y 0

for i in {1..10}
do
    random_x=$((RANDOM%9-4))
    random_y=$((RANDOM%19-9))

    ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "{name: box_$i}" > /dev/null
    ros2 run gazebo_ros spawn_entity.py -file $SCRIPT_DIR/models/box.sdf  -entity box_$i -x $random_x -y $random_y
done
