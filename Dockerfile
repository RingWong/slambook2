# 改进方案：
# From git AS a, From ...
# 多个基础镜像组合，拷贝build完后的库到最早的ubuntu镜像

# 本项目依赖环境的完整Docker镜像
FROM ubuntu:18.04

WORKDIR /slambook2

COPY . /slambook2

# 更新apt源地址，并安装Eigen库
# Eigen库默认安装在/usr/include/eigen3中
RUN apt install -y software-properties-common \
    && add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" \
    && sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && apt update \ 
    && apt install -y libeigen3-dev cmake build-essential libgl1-mesa-dev \
    libglew-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev \
    libsuitesparse-dev qtdeclarative5-dev qt5-qmake libqglviewer-dev-qt5 \
    libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev \
    && apt clean

# Pangolin
# https://github.com/stevenlovegrove/Pangolin
RUN cd /slambook2/3rdparty/Pangolin && mkdir build && cd build && cmake .. && cmake --build .

# Sophus
# https://github.com/strasdat/Sophus
RUN cd /slambook2/3rdparty/Sophus && mkdir build && cd build && cmake .. && make && make install

# Opencv
# apt-get install software-properties-common
# add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
# apt update
# apt-get install libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
# RUN cd /slambook2/3rdparty && git clone -b 3.4 https://github.com/opencv/opencv.git \
#     && git clone -b 3.4 https://github.com/opencv/opencv_contrib.git \
#     && cd opencv \
#     && cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=/slambook2/3rdparty/opencv/opencv_contrib-3.4.13/modules/ ..

RUN cd /slambook2/3rdparty/opencv && tar -zxf opencv-3.4.13.tar.gz opencv_contrib-3.4.13.tar.gz \
    && cd /slambook2/3rdparty/opencv/opencv-3.4.13 && mkdir build && cd build \
    && cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=/slambook2/3rdparty/opencv/opencv_contrib-3.4.13/modules/ .. \
    && make && make install


# ceres-solver
# https://github.com/ceres-solver/ceres-solver
# apt-get install libgoogle-glog-dev libgflags-dev libatlas-base-dev
RUN cd /slambook2/3rdparty/ceres-solver && mkdir build && cd build && cmake .. && make && make install

# g2o
# https://github.com/RainerKuemmerle/g2o
RUN cd /slambook2/3rdparty/g2o && mkdir build && cd build && cmake .. && make && make install

# googletest
# https://github.com/google/googletest
RUN cd /slambook2/3rdparty/googletest && mkdir build && cd build && cmake .. && make && make install

# DBoW3
# https://github.com/rmsalinas/DBow3
RUN cd /slambook2/3rdparty/DBoW3 && mkdir build && cd build && cmake .. && make && make install
