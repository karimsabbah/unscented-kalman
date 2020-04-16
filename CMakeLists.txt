cmake_minimum_required(VERSION 3.1)
project(UKF)

#set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

MESSAGE("EIGEN3_INCLUDE_DIRS = ${EIGEN3_INCLUDE_DIRS}") 

#Can manually add the sources using the set command as follows:
#set(SOURCES src/mainapp.cpp src/Student.cpp)

#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/*.cpp")
include_directories("src")

# set compiler options here
add_compile_options(-march=native -O1)

add_executable(ukf ${SOURCES})
target_link_libraries(ukf Eigen3::Eigen)

find_package(PythonLibs 2.7)
target_include_directories(ukf PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(ukf ${PYTHON_LIBRARIES})
