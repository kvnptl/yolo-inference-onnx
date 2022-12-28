cmake_minimum_required(VERSION 3.5.1)
project (yolo_inference_onnx)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4.6.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(hello src/helloworld.cpp)
target_link_libraries(hello ${OpenCV_LIBS})

add_executable(opencv_version src/opencv_version.cpp)
target_link_libraries(opencv_version ${OpenCV_LIBS})

add_executable(yolo_inference src/yolo_inference.cpp)
target_link_libraries(yolo_inference ${OpenCV_LIBS})

add_executable(yolo_inference_atwork src/yolo_inference_atwork.cpp)
target_link_libraries(yolo_inference_atwork ${OpenCV_LIBS})
