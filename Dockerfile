FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04


# disable terminal interaction for apt
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt install libcudnn8=8.5.0.*-1+cuda11.7  libcudnn8-dev=8.5.0.*-1+cuda11.7 && \
	cp /usr/include/cudnn_version.h /usr/local/cuda/include && \
	cp /usr/include/cudnn.h /usr/local/cuda/include && \
	rm -rf /var/lib/apt/lists/*

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        libglib2.0-0

# Fundamentals
RUN apt-get update && apt-get install -y \
        bash-completion \
        build-essential \
        clang-format \
        cmake \
        curl \
        git \
        gnupg2 \
        locales \
        lsb-release \
        rsync \
        software-properties-common \
        wget \
        vim \
        unzip \
        mlocate \
	libgoogle-glog-dev \
        && rm -rf /var/lib/apt/lists/*

# Install libtorch
RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# Python basics
RUN apt-get update && apt-get install -y \
        python3-pytest-cov \
        python3-setuptools \
        && rm -rf /var/lib/apt/lists/*
        
RUN python3 -m pip install pip==22.0.2
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Download detection config files
RUN wget -O model_cfg.py "https://cloud.dfki.de/owncloud/index.php/s/jRiZYbmJNcMdYY2/download"
RUN wget -O vision_guided_checkpoint.pth "https://cloud.dfki.de/owncloud/index.php/s/jRiZYbmJNcMdYY2/download"

        
# Setup ROS2 Foxy
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'      

RUN apt-get update && apt-get install -y \
        python3-colcon-common-extensions \
        python3-rosdep \
        ros-foxy-desktop python3-argcomplete \
        ros-foxy-ros-base python3-argcomplete \
        ros-dev-tools \
        ros-foxy-rqt* \
	ros-foxy-vision-msgs \
        && rm -rf /var/lib/apt/lists/*

RUN rosdep init
RUN rosdep update

# Source ROS 2 and set up the entrypoint
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/foxy/setup.bash && exec \"$@\"", "--"]
CMD ["bash"]
