#pragma once

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>
#include <vector>
#include "contact_parameters.hpp"

struct YamlLoader
{
    Eigen::VectorXd w_pos_body;
    Eigen::VectorXd w_pos_leg;
    Eigen::VectorXd w_vel_body;
    Eigen::VectorXd w_vel_leg;
    Eigen::VectorXd w_pos_body_term;
    Eigen::VectorXd w_pos_leg_term;
    Eigen::VectorXd w_vel_body_term;
    Eigen::VectorXd w_vel_leg_term;
    Eigen::VectorXd w_u_leg;
    double w_foot_slip_clearance;

    int nsteps;
    double timestep;
    int max_iter;

    ContactParams<double> real_contact_params;
    ContactParams<double> fake_contact_params;

    YamlLoader(const std::string &filepath)
    {
        try
        {
            YAML::Node config = YAML::LoadFile(filepath);

            w_pos_body = yamlSequenceToEigen(config["w_pos_body"]);
            w_pos_leg = yamlSequenceToEigen(config["w_pos_leg"]);
            w_vel_body = yamlSequenceToEigen(config["w_vel_body"]);
            w_vel_leg = yamlSequenceToEigen(config["w_vel_leg"]);
            w_pos_body_term = yamlSequenceToEigen(config["w_pos_body_term"]);
            w_pos_leg_term = yamlSequenceToEigen(config["w_pos_leg_term"]);
            w_vel_body_term = yamlSequenceToEigen(config["w_vel_body_term"]);
            w_vel_leg_term = yamlSequenceToEigen(config["w_vel_leg_term"]);
            w_u_leg = yamlSequenceToEigen(config["w_u_leg"]);
            w_foot_slip_clearance = config["w_foot_slip_clearance"].as<double>();

            nsteps = config["nsteps"].as<int>();
            timestep = config["timestep"].as<double>();
            max_iter = config["max_iter"].as<int>();

            // 读取 real_contact 二级结构
            YAML::Node real_contact = config["real_contact"];
            real_contact_params.contact_stiffness = real_contact["contact_stiffness"].as<double>();
            real_contact_params.smoothing_factor = real_contact["smoothing_factor"].as<double>();
            real_contact_params.dissipation_velocity = real_contact["dissipation_velocity"].as<double>();
            real_contact_params.stiction_velocity = real_contact["stiction_velocity"].as<double>();
            real_contact_params.friction_coefficient = real_contact["friction_coefficient"].as<double>();

            // 读取 fake_contact 二级结构
            YAML::Node fake_contact = config["fake_contact"];
            fake_contact_params.contact_stiffness = fake_contact["contact_stiffness"].as<double>();
            fake_contact_params.smoothing_factor = fake_contact["smoothing_factor"].as<double>();
            fake_contact_params.dissipation_velocity = fake_contact["dissipation_velocity"].as<double>();
            fake_contact_params.stiction_velocity = fake_contact["stiction_velocity"].as<double>();
            fake_contact_params.friction_coefficient = fake_contact["friction_coefficient"].as<double>();

            std::cout << "w_pos_body: " << w_pos_body.transpose() << std::endl;
            std::cout << "w_pos_leg: " << w_pos_leg.transpose() << std::endl;
            std::cout << "w_vel_body: " << w_vel_body.transpose() << std::endl;
            std::cout << "w_vel_leg: " << w_vel_leg.transpose() << std::endl;
            std::cout << "w_pos_body_term: " << w_pos_body_term.transpose() << std::endl;
            std::cout << "w_pos_leg_term: " << w_pos_leg_term.transpose() << std::endl;
            std::cout << "w_vel_body_term: " << w_vel_body_term.transpose() << std::endl;
            std::cout << "w_vel_leg_term: " << w_vel_leg_term.transpose() << std::endl;
            std::cout << "w_u_leg: " << w_u_leg.transpose() << std::endl;
            std::cout << "w_foot_slip_clearance: " << w_foot_slip_clearance << std::endl;

            std::cout << "nsteps: " << nsteps << std::endl;
            std::cout << "timestep: " << timestep << std::endl;
            std::cout << "max_iter: " << max_iter << std::endl;

            std::cout << "real_contact_params.contact_stiffness: " << real_contact_params.contact_stiffness << std::endl;
            std::cout << "real_contact_params.smoothing_factor: " << real_contact_params.smoothing_factor << std::endl;
            std::cout << "real_contact_params.dissipation_velocity: " << real_contact_params.dissipation_velocity << std::endl;
            std::cout << "real_contact_params.stiction_velocity: " << real_contact_params.stiction_velocity << std::endl;
            std::cout << "real_contact_params.friction_coefficient: " << real_contact_params.friction_coefficient << std::endl;
            std::cout << "fake_contact_params.contact_stiffness: " << fake_contact_params.contact_stiffness << std::endl;
            std::cout << "fake_contact_params.smoothing_factor: " << fake_contact_params.smoothing_factor << std::endl;
            std::cout << "fake_contact_params.dissipation_velocity: " << fake_contact_params.dissipation_velocity << std::endl;
            std::cout << "fake_contact_params.stiction_velocity: " << fake_contact_params.stiction_velocity << std::endl;
            std::cout << "fake_contact_params.friction_coefficient: " << fake_contact_params.friction_coefficient << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error loading YAML file: " << e.what() << std::endl;
        }
    }

private:
    /**
     * 将YAML序列转换为Eigen::VectorXd
     * @param node YAML序列节点
     * @return 包含序列值的Eigen向量
     */
    Eigen::VectorXd yamlSequenceToEigen(const YAML::Node &node)
    {
        if (!node.IsSequence())
        {
            throw std::runtime_error("YAML node is not a sequence");
        }

        Eigen::VectorXd vec(node.size());
        for (size_t i = 0; i < node.size(); ++i)
        {
            vec(i) = node[i].as<double>();
        }
        return vec;
    }
};