#!/usr/bin/env bash

source /opt/ros/noetic/setup.bash
cd ~/Documents/MTP/code/catkin_ws || exit
source src/venv/bin/activate
source devel/setup.bash

roslaunch diff_planner gazebo.launch wandb_file:=dl_rob/diffusion/5t0sxiys