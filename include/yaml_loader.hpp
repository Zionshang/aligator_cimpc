#pragma once

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>
#include <vector>

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
    int nsteps;
    double timestep;

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

            nsteps = config["nsteps"].as<int>();
            timestep = config["timestep"].as<double>();

            std::cout << "w_pos_body: " << w_pos_body.transpose() << std::endl;
            std::cout << "w_pos_leg: " << w_pos_leg.transpose() << std::endl;
            std::cout << "w_vel_body: " << w_vel_body.transpose() << std::endl;
            std::cout << "w_vel_leg: " << w_vel_leg.transpose() << std::endl;
            std::cout << "w_pos_body_term: " << w_pos_body_term.transpose() << std::endl;
            std::cout << "w_pos_leg_term: " << w_pos_leg_term.transpose() << std::endl;
            std::cout << "w_vel_body_term: " << w_vel_body_term.transpose() << std::endl;
            std::cout << "w_vel_leg_term: " << w_vel_leg_term.transpose() << std::endl;
            std::cout << "w_u_leg: " << w_u_leg.transpose() << std::endl;
            std::cout << "nsteps: " << nsteps << std::endl;
            std::cout << "timestep: " << timestep << std::endl;
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