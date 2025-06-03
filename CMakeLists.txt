cmake_minimum_required(VERSION 3.5.0)
project(aligator_cimpc VERSION 0.1.0 LANGUAGES C CXX)

find_package(aligator REQUIRED)
find_package(pinocchio REQUIRED)
find_package(proxsuite REQUIRED)

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
add_executable(webots_sim
    src/main.cpp
    src/contact_fwd_dynamics.cpp
    src/contact_force.cpp
    src/webots_interface.cpp
    src/foot_slip_clearance_cost.cpp
    src/interpolator.cpp
    src/symmetric_control_residual.cpp)
target_link_directories(webots_sim PRIVATE ${WEBOTS_HOME}/lib/controller)
set(WEBOTSLIBRARIES m ${CMAKE_SHARED_LIBRARY_PREFIX}Controller${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_SHARED_LIBRARY_PREFIX}CppController${CMAKE_SHARED_LIBRARY_SUFFIX})
target_include_directories(webots_sim PRIVATE ${WEBOTS_HOME}/include/controller/cpp)
target_link_libraries(webots_sim aligator::aligator
    pinocchio::pinocchio
    ${WEBOTSLIBRARIES}
    yaml-cpp
    proxsuite::proxsuite
)

add_executable(offline_fwd_dyn_test test/offline_fwd_dyn_test.cpp
    src/contact_force.cpp
    src/contact_fwd_dynamics.cpp
    src/foot_slip_clearance_cost.cpp
    src/symmetric_control_residual.cpp)

target_link_libraries(offline_fwd_dyn_test aligator::aligator
    pinocchio::pinocchio
    yaml-cpp)


add_executable(interpolator_test test/interpolator_test.cpp
    src/interpolator.cpp)

target_link_libraries(interpolator_test pinocchio::pinocchio)

add_executable(cost_comparation test/cost_comparation.cpp
    src/contact_force.cpp
    src/contact_fwd_dynamics.cpp
    src/foot_slip_clearance_cost.cpp
    src/symmetric_control_residual.cpp)

target_link_libraries(cost_comparation aligator::aligator
    pinocchio::pinocchio
    yaml-cpp)
