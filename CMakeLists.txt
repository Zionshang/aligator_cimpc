cmake_minimum_required(VERSION 3.5.0)
project(aligator_cimpc VERSION 0.1.0 LANGUAGES C CXX)

find_package(aligator REQUIRED)
find_package(pinocchio REQUIRED)
find_package(proxsuite REQUIRED)
find_package(example-robot-data REQUIRED)

include_directories(include)

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

# webots
set(WEBOTS_HOME "/usr/local/webots")
set(WEBOTS_LIB_DIR "${WEBOTS_HOME}/lib/controller")
set(WEBOTS_INCLUDE_DIR "${WEBOTS_HOME}/include/controller/cpp")
set(WEBOTS_LIBS
    m
    ${CMAKE_SHARED_LIBRARY_PREFIX}Controller${CMAKE_SHARED_LIBRARY_SUFFIX}
    ${CMAKE_SHARED_LIBRARY_PREFIX}CppController${CMAKE_SHARED_LIBRARY_SUFFIX}
)

add_executable(webots_sim
    src/main.cpp
    src/contact_fwd_dynamics.cpp
    src/contact_force.cpp
    src/webots_interface.cpp
    src/foot_slip_clearance_cost.cpp
    src/interpolator.cpp
    src/contact_inv_dynamics_residual.cpp
    src/contact_inv_dynamics_residual2.cpp)
target_include_directories(webots_sim PRIVATE ${WEBOTS_INCLUDE_DIR})
target_link_directories(webots_sim PRIVATE ${WEBOTS_LIB_DIR})
target_link_libraries(webots_sim aligator::aligator
    pinocchio::pinocchio
    ${WEBOTS_LIBS}
    yaml-cpp
    proxsuite::proxsuite
    example-robot-data::example-robot-data
)

add_executable(contact_inv_dynamics_test
    test/contact_inv_dynamics_test.cpp
    src/contact_fwd_dynamics.cpp
    src/contact_inv_dynamics_residual.cpp
    src/contact_force.cpp)
target_link_libraries(contact_inv_dynamics_test aligator::aligator
    pinocchio::pinocchio)

add_executable(offline_inv_dyn_test test/offline_inv_dyn_test.cpp
    src/contact_force.cpp
    src/contact_inv_dynamics_residual.cpp
    src/contact_inv_dynamics_residual2.cpp
    src/foot_slip_clearance_cost.cpp
    src/symmetric_control_residual.cpp)

target_link_libraries(offline_inv_dyn_test aligator::aligator
    pinocchio::pinocchio
    yaml-cpp
    example-robot-data::example-robot-data)

add_executable(offline_fwd_dyn_test test/offline_fwd_dyn_test.cpp
    src/contact_force.cpp
    src/contact_fwd_dynamics.cpp
    src/foot_slip_clearance_cost.cpp)

target_link_libraries(offline_fwd_dyn_test aligator::aligator
    pinocchio::pinocchio
    yaml-cpp)

add_executable(inv_dynamics_test test/inv_dynamics_test.cpp)
target_link_libraries(inv_dynamics_test pinocchio::pinocchio)

add_executable(symmetric_control_residual_test
    test/symmetric_control_residual_test.cpp
    src/symmetric_control_residual.cpp)
target_link_libraries(symmetric_control_residual_test
    aligator::aligator
    example-robot-data::example-robot-data)
