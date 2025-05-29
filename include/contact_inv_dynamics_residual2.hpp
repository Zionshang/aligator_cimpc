#pragma once

#include <aligator/core/function-abstract.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/autodiff/cppad.hpp>

#include "contact_force.hpp"

using StageFunction = aligator::StageFunctionTpl<double>;
using StageFunctionData = aligator::StageFunctionDataTpl<double>;
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct ContactInvDynamicsResidual2 : StageFunction
{
    ContactInvDynamicsResidual2(int ndx, const Model &model, MatrixXd actuation, double timestep,
                               ContactParams<double> contact_params);

    void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                  StageFunctionData &data) const;

    void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                          StageFunctionData &data) const;

    std::shared_ptr<StageFunctionData> createData() const;

    pinocchio::Model model_; // 存储着pinocchio模型
    MatrixXd actuation_matrix_;
    ContactParams<double> contact_params_;
    double timestep_;
};

struct ContactInvDynamicsResidualData2 : StageFunctionData
{
    ContactInvDynamicsResidualData2(const ContactInvDynamicsResidual2 &resdl);

    VectorXd tau_;
    pinocchio::Data data_;
    MatrixXd dtau_dtaua_; // 广义力矩对驱动力矩的偏导
    CppAD::ADFun<double> ad_inv_dynamics_;
    std::vector<Vector3d> contact_forces_; // 仅仅用于记录数据便于debug
};