cmake_minimum_required(VERSION 3.5.0)
project(aligator_cimpc VERSION 0.1.0 LANGUAGES C CXX)

find_package(aligator REQUIRED)
find_package(pinocchio REQUIRED)

include_directories(include)

add_executable(offline_test test/offline_test.cpp
    src/contact_force.cpp
    src/contact_fwd_dynamics.cpp
    src/foot_slip_clearance_cost.cpp)

target_link_libraries(offline_test aligator::aligator
    pinocchio::pinocchio
    yaml-cpp)


add_executable(cppad_test
    test/cppad_test.cpp
    src/contact_force.cpp)
target_link_libraries(cppad_test aligator::aligator
    pinocchio::pinocchio)

add_executable(contact_test test/contact_test.cpp)
target_link_libraries(contact_test aligator::aligator
    pinocchio::pinocchio)

add_executable(contact_fwd_dynamics_test
    test/contact_fwd_dynamics_test.cpp
    src/contact_fwd_dynamics.cpp
    src/contact_force.cpp)
target_link_libraries(contact_fwd_dynamics_test aligator::aligator
    pinocchio::pinocchio)
