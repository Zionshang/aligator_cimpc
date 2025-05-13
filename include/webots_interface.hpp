#pragma once
#include <webots/Motor.hpp>
#include <webots/PositionSensor.hpp>
#include <webots/InertialUnit.hpp>
#include <webots/Gyro.hpp>
#include <webots/Supervisor.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <webots/Keyboard.hpp>
#include <webots/Joystick.hpp>

class WebotsInterface
{
public:
    WebotsInterface();
    ~WebotsInterface();
    void recvState(Eigen::VectorXd &state_vector); // second
    void sendCmd(const Eigen::VectorXd &tau);
    bool isRunning();
    double current_time() { return current_time_; }
    void resetSim() { supervisor_->simulationReset(); }
    double timestep() { return double(time_step_) / 1000; }

private:
    void initRecv();
    void initSend();

    int time_step_;
    double current_time_;

    Eigen::VectorXd last_q_;

    // webots interface
    webots::Supervisor *supervisor_;
    webots::Node *robot_node_;
    std::vector<webots::Motor *> joint_motor_;
    std::vector<webots::PositionSensor *> joint_sensor_;
    webots::InertialUnit *imu_;
    webots::Gyro *gyro_;
    webots::Keyboard *keyboard_;

    std::string robot_name_ = "MiniCheetah";
    std::string imu_name_ = "trunk_imu_inertial";
    std::string gyro_name_ = "trunk_imu_gyro";
    std::string accelerometer_name_ = "accelerometer";
    std::vector<std::string> joint_motor_name_ = {"torso_to_abduct_fl_j", "abduct_fl_to_thigh_fl_j", "thigh_fl_to_knee_fl_j",
                                                  "torso_to_abduct_fr_j", "abduct_fr_to_thigh_fr_j", "thigh_fr_to_knee_fr_j",
                                                  "torso_to_abduct_hl_j", "abduct_hl_to_thigh_hl_j", "thigh_hl_to_knee_hl_j",
                                                  "torso_to_abduct_hr_j", "abduct_hr_to_thigh_hr_j", "thigh_hr_to_knee_hr_j"};
    std::vector<std::string> joint_sensor_name_ = {"torso_to_abduct_fl_j_sensor", "abduct_fl_to_thigh_fl_j_sensor", "thigh_fl_to_knee_fl_j_sensor",
                                                   "torso_to_abduct_fr_j_sensor", "abduct_fr_to_thigh_fr_j_sensor", "thigh_fr_to_knee_fr_j_sensor",
                                                   "torso_to_abduct_hl_j_sensor", "abduct_hl_to_thigh_hl_j_sensor", "thigh_hl_to_knee_hl_j_sensor",
                                                   "torso_to_abduct_hr_j_sensor", "abduct_hr_to_thigh_hr_j_sensor", "thigh_hr_to_knee_hr_j_sensor"};

    int num_joints_;
    int key_, last_key_;
};