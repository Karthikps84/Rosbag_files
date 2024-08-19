# Environment Setup

This repository provides a Docker environment configured with ROS2 and ML libraries. 
It includes installation instructions, building, and running the Docker container, as well as sourcing ROS2 
and running a Python script within the container.

## Table of Contents
- [Installation](#installation)
- [Building the Docker Image](#building-the-docker-image)
- [Running the Docker Container](#running-the-docker-container)
- [Required Modules for Host Machine](#required-modules-for-host-machine)
- [Running Python Script](#running-python-script)
- [Published ROS Topics](#published-ros-topics)
- [Common Issues](#common-issues)

## Installation (Pre-requisites)

- Ensure the Docker is installed on your system. It can be downloaded from [here](https://docs.docker.com/get-docker/).
- ROS2 Humble [Link](https://docs.ros.org/en/humble/Installation.html).
- NVIDIA container [Link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Building the Docker Image

Clone this repository, navigate to the directory and build the docker image:
```
git clone https://github.com/eureka-HERON/situational_awareness.git
```

```
cd situational_awareness

docker build -t heron_env:latest .
```

## Running the Docker Container
Since we are working with ROS2 Humble on this project, we assume it has already been installed. Hence, we need to source the docker container to the ROS.
```
docker run -it --net=host  --rm -v /opt/ros/humble:/opt/ros/humble --name container --gpus all heron_env:latest
```

## Required Modules for Host Machine:

In order to communicate between Docker and the host machine, execute the following commands on the host machine:
```
sudo apt install ros-humble-rmw-cyclonedds-cpp
```
```
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

## Running Python Script
```
python3 publish.py
```
Here are the arguments that can be modified in the script:
```
options:
  -h, --help            show this help message and exit
  --distance_threshold DISTANCE_THRESHOLD
                        Distance threshold for object detection. Default is set to 50 CM.
  --detection_confidence_score DETECTION_CONFIDENCE_SCORE
                        Detection confidence score to filter detections. Default is 0.5.
  --num_cameras NUM_CAMERAS
                        Number of cameras to subscribe. Same for number of CPU Cores to be utilized. Default is 4.
```

## ROS Topics
### Subscriber Topics:
The script by defaults subscribes to following topics with **same Camera name and Camera name space**:
```
camera_topics = {
    'cam_hand': (
        '/cam_hand/cam_hand/color/image_raw', '/cam_hand/cam_hand/depth/image_rect_raw', '/cam_hand/cam_hand/depth/camera_info'),
    'cam_center': (
        '/cam_center/cam_center/color/image_raw', '/cam_center/cam_center/depth/image_rect_raw', '/cam_center/cam_center/depth/camera_info'),
    'cam_right': (
        '/cam_right/cam_right/color/image_raw', '/cam_right/cam_right/depth/image_rect_raw', '/cam_right/cam_right/depth/camera_info'),
    'cam_left': (
        '/cam_left/cam_left/color/image_raw', '/cam_left/cam_left/depth/image_rect_raw', '/cam_left/cam_left/depth/camera_info')
}
```
### Publisher Topics:
Once the script starts running, whenever the obstacle is identified, for each obstacle, data will be published on two topics:

1- /distance (This is a std String Message. It gives the distance value in meters and also the camera name)
```
data: 'From cam_center, Distance: 0.33229169249534607 meters'
```
2- /boundingbox_3D (This is a boundingbox3D Vision message.)
```
center:
  position:
    x: 0.19711093139648436
    y: -0.048634571075439455
    z: 0.176
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0
size:
  x: 0.32192441826199397
  y: 0.29874923108344725
  z: 0.1770833432674408
---
```



## Common Issues:
One (unlikely) issue could be with mmcv version mismatch. In this case, mmcv needs to be updated.
```
mim uninstall mmcv

mim install mmcv==2.1.0
```
