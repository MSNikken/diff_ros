# Denoising Diffusion Planner: Learning Complex Paths from Low-Quality Demonstrations - ROS packages

This repo contains the ROS implementation for experiments with a Franka Emika Panda robot. 

## Installation
Clone this repo and place the packages in your `catkin_ws`. Build the packages and install the python packages:
```
python -m venv /path/to/new/virtual/environment
pip install -r requirements.txt
```

## Usage
The `diff_planner` package contains a node that can load parameters from a `pt`file or directly from a `wandb` save. These are settable as ROS params.
```
rosparam set diff_planner/wandb_file path/to/wandb/run
rosparam set diff_planner/pt_file path/to/pt_file.pt
```
Also, runtime inference parameters of the DDPM can be set in a `.yml` file and setting the file as ROS param.
```
rosparam set diff_planner/config/file path/to/config.yml
```
Run the planner node using:
```
rosrun diff_planner diff_planner
```
The 'diff_planner' node publishes setpoints that can be used by the `cartesian_impedance_example_controller` provided by Franka Robotics.
Note that `planners.py` contains safety features and planning settings, liking clamping the outputs or setting the planning horizon, that can currently only be adjusted by editing `planners.py`.
