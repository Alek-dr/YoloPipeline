cmake_minimum_required(VERSION 3.23)
project(YoloPipeline)

add_definitions(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)

#/<----------------Add Cuda----------------->
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

#/<----------------Add Tensorrt----------------->
include_directories(/usr/local/TensorRT-8.6.0.12/include)
link_directories(/usr/local/TensorRT-8.6.0.12/lib)

#/<----------------Add OpenCV----------------->
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(YoloPipeline main.cpp src/yolo.h src/yolo.cpp)

target_link_libraries(YoloPipeline nvinfer)
target_link_libraries(YoloPipeline nvinfer_plugin)
target_link_libraries(YoloPipeline cudart)
target_link_libraries(YoloPipeline ${OpenCV_LIBS})

add_definitions(-O2 -pthread)
