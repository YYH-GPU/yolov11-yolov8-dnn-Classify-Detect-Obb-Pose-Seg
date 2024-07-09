cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(YOLO)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 包含头文件目录
include_directories(include)

# 查找src文件夹下的所有源文件
file(GLOB SOURCES "src/*.cpp")

# 添加可执行文件
add_executable(MyProject main.cpp ${SOURCES})

# 链接OpenCV库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(MyProject ${OpenCV_LIBS})
