cmake_minimum_required(VERSION 3.5.0)
project(aligator_cimpc VERSION 0.1.0 LANGUAGES C CXX)

find_package(aligator REQUIRED)
find_package(pinocchio REQUIRED)

add_executable(offline_test offline_test.cpp CompliantContactFwdDynamics.cpp)

target_link_libraries(offline_test aligator::aligator)
target_link_libraries(offline_test pinocchio::pinocchio)
