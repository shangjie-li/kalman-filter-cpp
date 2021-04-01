cmake_minimum_required(VERSION 2.8.4)
project(kalman-filter-cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(EIGEN_INC_DIR /usr/include/eigen3/)
include_directories(${EIGEN_INC_DIR})

find_package(OpenCV REQUIRED)

add_executable(multiple_target_tracking multiple_target_tracking.cpp)
target_link_libraries(multiple_target_tracking ${OpenCV_LIBS})
