# SOURCE: https://rodosingh.medium.com/using-cmake-to-build-and-install-opencv-for-python-and-c-in-ubuntu-20-04-6c5881eebd9a

# Modified by: Kevin Patel
# Date: 14-May-2023

# https://askubuntu.com/questions/484718/how-to-make-a-file-executable
# bash script for installing OpenCV from Github source...
sudo apt update && sudo apt upgrade
sudo apt install -y build-essential cmake git unzip pkg-config make libjpeg-dev libpng-dev libtiff-dev libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev
# sudo apt-get install -y python3.8-dev python3-numpy libtbb2 libtbb-dev
sudo apt-get install -y libtbb2 libtbb-dev
# libdc1394-22-dev libavresample-dev not available in ubuntu 22.04
sudo apt install -y libeigen3-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev sphinx-common libtbb-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libavutil-dev libavfilter-dev 
mkdir ~/opencv_build && cd ~/opencv_build
git clone https://github.com/opencv/opencv
git clone https://github.com/opencv/opencv_contrib
cd ~/opencv_build/opencv
mkdir -p build && cd build
cmake -D WITH_CUDA=OFF -D BUILD_TIFF=ON -D BUILD_opencv_java=OFF -D WITH_OPENGL=ON -D WITH_OPENCL=ON -D WITH_IPP=ON -D WITH_TBB=ON -D WITH_EIGEN=ON -D WITH_V4L=ON -D WITH_VTK=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D CMAKE_BUILD_TYPE=RELEASE -D BUILD_opencv_python2=OFF -D CMAKE_INSTALL_PREFIX=/usr/local -D PYTHON3_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") -D PYTHON3_PACKAGES_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))") -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_ENABLE_NONFREE=ON -D OPENCV_GENERATE_PKGCONFIG=ON -D PYTHON3_EXECUTABLE=$(which python3) -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules -D BUILD_EXAMPLES=ON ..

# make -j<no. of cores>
make -j12
sudo make install
sudo ldconfig
# To verify the installation, type the following commands and you should see the OpenCV version.
# C++ bindings:
pkg-config --modversion opencv4
# Python bindings:
# cd ~/miniconda3/envs/rodo_env/lib/python3.8/site-packages/
# ln -s /usr/local/lib/python3.8/site-packages/cv2/python-3.8/cv2.cpython-38-x86_64-linux-gnu.so cv2.so
# python3 -c "import cv2; print(cv2.__version__)"s