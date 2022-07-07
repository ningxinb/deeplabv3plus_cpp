cmake_minimum_required(VERSION 3.22)
project(cpptest)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)
link_directories(/home/ning/software/tensorflow-2.3.0/bazel-bin/tensorflow)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 REQUIRED)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()

include_directories(
        /home/ning/software/tensorflow-2.3.0
        /home/ning/software/tensorflow-2.3.0/bazel-bin
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)
add_executable(cpptest main.cpp)
target_link_libraries(cpptest
        ${EIGEN3_LIBS}
        ${OpenCV_LIBS}
        /home/ning/software/tensorflow-2.3.0/bazel-bin/tensorflow/libtensorflow_cc.so
        )
