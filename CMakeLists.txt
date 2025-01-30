cmake_minimum_required(VERSION 3.5.0)
project(aligator_cimpc VERSION 0.1.0 LANGUAGES C CXX)

find_package(aligator REQUIRED)
find_package(pinocchio REQUIRED)

add_executable(aligator_cimpc main.cpp CompliantContactFwdDynamics.cpp)

target_link_libraries(aligator_cimpc aligator::aligator)
target_link_libraries(aligator_cimpc pinocchio::pinocchio)
